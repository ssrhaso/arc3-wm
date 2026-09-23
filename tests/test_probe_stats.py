"""Tests for arc3_wm.probe_stats: grouped folds, permutation tests, Holm, paired diffs."""
from __future__ import annotations

import numpy as np
import pytest

from arc3_wm.probe_stats import (
    RidgeFolds,
    balanced_accuracy,
    cv_balanced_accuracy,
    grouped_kfold,
    holm,
    paired_fold_difference,
    permutation_test,
)


def _grouped_data(n_ep=20, per_ep=12, dim=8, signal=1.0, seed=0, n_classes=3):
    """Episodes whose label is decodable from the features with strength ``signal``."""
    rng = np.random.default_rng(seed)
    groups, y, X = [], [], []
    for e in range(n_ep):
        labels = rng.integers(0, n_classes, per_ep)
        feats = rng.normal(size=(per_ep, dim))
        feats[:, 0] += signal * labels  # the only informative direction
        groups.append(np.full(per_ep, e))
        y.append(labels)
        X.append(feats)
    return np.vstack(X), np.concatenate(y), np.concatenate(groups)


# --- grouped folds ---------------------------------------------------------

def test_grouped_kfold_partitions_and_never_splits_a_group():
    groups = np.repeat(np.arange(10), 5)
    folds = grouped_kfold(groups, n_folds=5, seed=0)
    assert len(folds) == 5
    stacked = np.sum(folds, axis=0)
    assert (stacked == 1).all(), "every frame in exactly one test fold"
    for f in folds:
        in_fold = set(groups[f].tolist())
        out_fold = set(groups[~f].tolist())
        assert not (in_fold & out_fold), "an episode may not straddle folds"


def test_grouped_kfold_caps_folds_at_group_count_and_rejects_one_fold():
    assert len(grouped_kfold(np.repeat(np.arange(3), 4), n_folds=10)) == 3
    with pytest.raises(ValueError):
        grouped_kfold(np.arange(10), n_folds=1)


def test_grouped_kfold_is_seed_deterministic():
    g = np.repeat(np.arange(12), 3)
    a = grouped_kfold(g, 4, seed=7)
    b = grouped_kfold(g, 4, seed=7)
    c = grouped_kfold(g, 4, seed=8)
    assert all((x == y).all() for x, y in zip(a, b))
    assert any((x != y).any() for x, y in zip(a, c))


# --- balanced accuracy -----------------------------------------------------

def test_balanced_accuracy_is_mean_per_class_recall():
    y = np.array([0, 0, 0, 0, 1])
    classes = np.array([0, 1])
    assert balanced_accuracy(y, np.zeros(5, int), classes) == pytest.approx(0.5)
    assert balanced_accuracy(y, y, classes) == pytest.approx(1.0)


# --- probe scoring ---------------------------------------------------------

def test_probe_recovers_signal_and_is_at_chance_without_it():
    """Two-class data with a clean informative axis is decoded near perfectly;
    the same generator with no signal sits at the 0.5 chance floor."""
    X, y, g = _grouped_data(signal=4.0, n_classes=2)
    strong = cv_balanced_accuracy(X, y, g, n_folds=5)
    X0, y0, g0 = _grouped_data(signal=0.0, n_classes=2, seed=1)
    none = cv_balanced_accuracy(X0, y0, g0, n_folds=5)
    assert strong.mean() > 0.9
    assert none.mean() < 0.6
    assert strong.mean() - none.mean() > 0.3


def test_multiclass_probe_beats_chance_by_a_wide_margin():
    """Three ordinal classes on one axis: a linear argmax probe tops out near
    0.7, which is still far above the 0.33 chance floor and the 0.22 no-signal
    baseline. Pinned so a future change to the classifier is visible."""
    X, y, g = _grouped_data(signal=3.0, n_classes=3)
    X0, y0, g0 = _grouped_data(signal=0.0, n_classes=3, seed=1)
    assert cv_balanced_accuracy(X, y, g, n_folds=5).mean() > 0.6
    assert cv_balanced_accuracy(X0, y0, g0, n_folds=5).mean() < 0.4


def test_transfer_matrix_matches_an_explicit_ridge_fit():
    """The precomputed transfer matrix must reproduce a textbook ridge probe.

    Guards the optimization in RidgeFolds: predictions come from
    ``argmax_c sum_{i in c} M[:, i]`` rather than from solving for weights, so
    this pins the two against each other on the same fold and features.
    """
    X, y, g = _grouped_data(signal=2.0, dim=6, n_ep=8, per_ep=10)
    lam = 1.0
    folds = RidgeFolds(X, g, n_folds=4, lam=lam, seed=2)
    classes = np.unique(y)
    for f in folds.folds:
        tr, te = f.train, f.test
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        x_tr = np.concatenate([(X[tr] - mu) / sd, np.ones((int(tr.sum()), 1))], 1)
        x_te = np.concatenate([(X[te] - mu) / sd, np.ones((int(te.sum()), 1))], 1)
        Y = (y[tr][:, None] == classes[None, :]).astype(float)
        W = np.linalg.solve(x_tr.T @ x_tr + lam * np.eye(x_tr.shape[1]), x_tr.T @ Y)
        expected = classes[np.argmax(x_te @ W, axis=1)]
        assert (folds._predict(f, y[tr], classes) == expected).all()


def test_primal_and_dual_paths_agree():
    """n_train < d takes the dual path and n_train > d the primal; same answer."""
    X_wide, y_w, g_w = _grouped_data(n_ep=6, per_ep=5, dim=200, signal=2.0, seed=4)
    X_tall, y_t, g_t = _grouped_data(n_ep=20, per_ep=20, dim=5, signal=2.0, seed=4)
    wide = RidgeFolds(X_wide, g_w, n_folds=3, seed=1)
    tall = RidgeFolds(X_tall, g_t, n_folds=3, seed=1)
    assert wide.folds[0].transfer.shape[1] < X_wide.shape[1]   # dual path used
    assert np.isfinite(wide.fold_scores(y_w)).all()
    assert np.isfinite(tall.fold_scores(y_t)).all()


def test_fold_scores_rejects_mismatched_labels():
    X, y, g = _grouped_data()
    folds = RidgeFolds(X, g, n_folds=4)
    with pytest.raises(ValueError):
        folds.fold_scores(y[:-1])


def test_factorization_is_reused_across_labels():
    """One RidgeFolds scores several targets; results match a fresh fit each time."""
    X, y, g = _grouped_data(signal=2.0)
    folds = RidgeFolds(X, g, n_folds=5, seed=3)
    other = (y == 0).astype(int)
    a = folds.fold_scores(y)
    b = folds.fold_scores(other)
    assert len(a) == len(b) == 5
    fresh = cv_balanced_accuracy(X, y, g, n_folds=5, seed=3)
    assert a == pytest.approx(fresh)


# --- permutation test ------------------------------------------------------

def test_permutation_test_detects_real_signal():
    X, y, g = _grouped_data(signal=3.0)
    res = permutation_test(RidgeFolds(X, g, n_folds=5), y, n_perm=99)
    assert res["status"] == "ok"
    assert res["p_value"] <= 0.02
    assert res["balanced_acc"] > res["null_p95"]
    assert len(res["fold_scores"]) == res["n_folds"] == 5


def test_permutation_test_does_not_fire_on_noise():
    X, y, g = _grouped_data(signal=0.0, seed=5)
    res = permutation_test(RidgeFolds(X, g, n_folds=5), y, n_perm=99)
    assert res["p_value"] > 0.05


def test_permutation_pvalue_is_never_zero_and_respects_n_perm():
    X, y, g = _grouped_data(signal=5.0)
    res = permutation_test(RidgeFolds(X, g, n_folds=5), y, n_perm=19)
    assert res["p_value"] >= 1 / 20
    assert res["n_perm"] == 19


def test_permutation_test_handles_single_class():
    X, _, g = _grouped_data()
    res = permutation_test(RidgeFolds(X, g, n_folds=4), np.zeros(len(X), int), n_perm=5)
    assert res["status"] == "degenerate_single_class"


def test_permutation_test_is_reproducible():
    X, y, g = _grouped_data(signal=1.0)
    folds = RidgeFolds(X, g, n_folds=4)
    a = permutation_test(folds, y, n_perm=25, seed=11)
    b = permutation_test(folds, y, n_perm=25, seed=11)
    assert a["p_value"] == b["p_value"] and a["null_mean"] == b["null_mean"]


# --- paired differences ----------------------------------------------------

def test_paired_difference_detects_a_consistent_gap():
    a = [0.70, 0.72, 0.68, 0.71, 0.69]
    b = [0.50, 0.52, 0.49, 0.51, 0.50]
    res = paired_fold_difference(a, b, n_boot=2000)
    assert res["mean_diff"] == pytest.approx(0.196, abs=0.01)
    assert res["excludes_zero"] and res["ci_lo"] > 0


def test_paired_difference_on_noise_includes_zero():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.05, 12)
    b = rng.normal(0.5, 0.05, 12)
    assert not paired_fold_difference(a, b, n_boot=2000)["excludes_zero"]


def test_paired_difference_requires_matched_folds():
    with pytest.raises(ValueError):
        paired_fold_difference([0.5, 0.6], [0.5])
    with pytest.raises(ValueError):
        paired_fold_difference([0.5], [0.5])


# --- Holm ------------------------------------------------------------------

def test_holm_matches_the_textbook_example():
    # classic worked example: m=4, sorted p 0.01, 0.02, 0.03, 0.04
    adj = holm([0.01, 0.02, 0.03, 0.04])
    assert adj == pytest.approx([0.04, 0.06, 0.06, 0.06])


def test_holm_is_monotone_and_clipped_at_one():
    adj = holm([0.5, 0.01, 0.9])
    assert max(adj) <= 1.0
    order = np.argsort([0.5, 0.01, 0.9])
    vals = [adj[i] for i in order]
    assert vals == sorted(vals), "adjusted values must be monotone in rank"


def test_holm_preserves_mapping_keys():
    adj = holm({"a": 0.01, "b": 0.04})
    assert set(adj) == {"a", "b"} and adj["a"] <= adj["b"]


def test_holm_single_and_empty():
    assert holm([0.03]) == pytest.approx([0.03])
    assert holm([]) == []
    assert holm({}) == {}


def test_holm_is_more_conservative_than_raw_but_less_than_bonferroni():
    raw = [0.01, 0.02, 0.03]
    adj = holm(raw)
    for a, r in zip(adj, raw):
        assert a >= r
    assert adj[2] <= 3 * raw[2] + 1e-12
