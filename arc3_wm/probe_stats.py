"""Statistics for the linear latent probes: grouped K-fold, permutation tests, Holm.

The Phase-6 probe scored decodability by repeating a random 70/30 episode split
20 times and calling a target decodable when the 2.5th percentile of the probe's
fold accuracies exceeded the 97.5th percentile of a control that permuted labels
once per fold. Those percentiles are not a confidence interval: the test sets
overlap heavily, so the spread understates uncertainty, and with one permutation
per fold the control is a single noisy draw. Nothing corrected for running the
same test over six games, several targets and three feature sets.

This module replaces that with three standard pieces:

* **Grouped K-fold.** Every episode appears in exactly one test fold, so the
  per-fold scores partition the data instead of overlapping.
* **Permutation test.** The whole grouped-K-fold statistic is recomputed on
  labels shuffled across all frames, ``n_perm`` times, giving
  ``p = (1 + #{perm >= observed}) / (1 + n_perm)`` (Ojala and Garriga, 2010).
  Shuffling breaks any relation between features and labels while leaving the
  grouped cross-validation untouched, so the null is "this feature set carries
  no linearly decodable information about this label".
* **Holm correction** across the whole family of tests, and paired per-fold
  differences for "is the latent better than the pixel baseline", which the
  earlier design could not answer because it compared two independent intervals.

Permutation testing a 2560-dimensional ridge probe is only affordable because
ridge regression to one-hot targets is linear in the labels: ``RidgeFolds``
precomputes the per-fold transfer matrix that maps training labels to test
predictions, so every extra target and every permutation costs a grouped column
sum rather than a fresh solve.

Numpy only, no sklearn, laptop-runnable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "grouped_kfold",
    "balanced_accuracy",
    "RidgeFolds",
    "cv_balanced_accuracy",
    "permutation_test",
    "paired_fold_difference",
    "holm",
]


def grouped_kfold(groups: np.ndarray, n_folds: int, seed: int = 0) -> list[np.ndarray]:
    """Test masks for ``n_folds`` folds, each whole group in exactly one fold.

    Groups (episodes) are shuffled then dealt round-robin, which keeps fold
    sizes close even when episodes differ in length.
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2; got {n_folds}")
    n_folds = min(n_folds, len(uniq))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    assignment = {int(uniq[o]): i % n_folds for i, o in enumerate(order)}
    fold_of = np.array([assignment[int(g)] for g in groups])
    return [fold_of == f for f in range(n_folds)]


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> float:
    """Mean per-class recall over the classes present in ``y_true``."""
    recalls = []
    for c in classes:
        m = y_true == c
        if m.sum() > 0:
            recalls.append(float((y_pred[m] == c).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


@dataclass
class _Fold:
    train: np.ndarray
    test: np.ndarray
    transfer: np.ndarray
    """``x_te (X'X + lambda I)^-1 x_tr'``, shape (n_test, n_train)."""


class RidgeFolds:
    """Grouped K-fold ridge probe with all label-independent work done once.

    Ridge regression to one-hot targets is linear in the labels, so the
    prediction on a fold's test rows is ``M Y`` where
    ``M = x_te (x_tr' x_tr + lambda I)^-1 x_tr'`` depends only on the features.
    Computing ``M`` once per fold turns each extra label vector, and therefore
    each permutation, into a grouped column sum over ``M``: the cost of a
    permutation test stops scaling with the 2048-dimensional latent and becomes
    negligible.

    ``M`` is built through the dual (kernel) form when the training set is
    smaller than the feature dimension, using the identity
    ``(X'X + lambda I)^-1 X' = X' (XX' + lambda I)^-1``, which is the cheaper
    factorization in exactly that regime.
    """

    def __init__(self, X: np.ndarray, groups: np.ndarray, *, n_folds: int = 10,
                 lam: float = 1.0, seed: int = 0) -> None:
        X = np.asarray(X, dtype=np.float64)
        self.n_samples = len(X)
        self.folds: list[_Fold] = []
        for test in grouped_kfold(groups, n_folds, seed):
            train = ~test
            n_tr, n_te = int(train.sum()), int(test.sum())
            if n_tr == 0 or n_te == 0:
                continue
            mu = X[train].mean(0)
            sd = X[train].std(0) + 1e-6
            x_tr = np.concatenate([(X[train] - mu) / sd, np.ones((n_tr, 1))], 1)
            x_te = np.concatenate([(X[test] - mu) / sd, np.ones((n_te, 1))], 1)
            d = x_tr.shape[1]
            if n_tr < d:
                gram = x_tr @ x_tr.T + lam * np.eye(n_tr)
                transfer = np.linalg.solve(gram.T, (x_te @ x_tr.T).T).T
            else:
                A = x_tr.T @ x_tr + lam * np.eye(d)
                transfer = x_te @ np.linalg.solve(A, x_tr.T)
            self.folds.append(_Fold(train, test, transfer))
        if not self.folds:
            raise ValueError("no usable folds")

    @staticmethod
    def _predict(fold: _Fold, y_tr: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """Ridge prediction as one matrix product: ``argmax(transfer @ onehot(y))``."""
        onehot = (y_tr[:, None] == classes[None, :]).astype(np.float64)
        return classes[np.argmax(fold.transfer @ onehot, axis=1)]

    def fold_scores(self, y: np.ndarray, classes: Optional[np.ndarray] = None) -> np.ndarray:
        """Balanced accuracy on each fold's held-out episodes."""
        y = np.asarray(y)
        if len(y) != self.n_samples:
            raise ValueError(f"y has {len(y)} rows, features have {self.n_samples}")
        classes = np.unique(y) if classes is None else classes
        out = []
        for f in self.folds:
            y_tr = y[f.train]
            if len(np.unique(y_tr)) < 2:
                continue
            out.append(balanced_accuracy(y[f.test], self._predict(f, y_tr, classes), classes))
        return np.asarray(out, dtype=float)


def cv_balanced_accuracy(X, y, groups, *, n_folds: int = 10, lam: float = 1.0,
                         seed: int = 0) -> np.ndarray:
    """Convenience wrapper: per-fold balanced accuracies for one feature set."""
    return RidgeFolds(X, groups, n_folds=n_folds, lam=lam, seed=seed).fold_scores(y)


def permutation_test(folds: RidgeFolds, y: np.ndarray, *, n_perm: int = 500,
                     seed: int = 0) -> dict:
    """Permutation test for "these features decode this label".

    The observed statistic is the mean per-fold balanced accuracy. Each
    permutation shuffles ``y`` across all frames and recomputes it. Reported
    ``p_value`` is the standard ``(1 + hits) / (1 + n_perm)`` estimator, which
    is never zero and is valid for finite ``n_perm``.
    """
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) < 2:
        return {"status": "degenerate_single_class", "n_classes": int(len(classes))}
    observed_folds = folds.fold_scores(y, classes)
    if len(observed_folds) < 2:
        return {"status": "too_few_folds", "n_folds": int(len(observed_folds))}
    observed = float(np.mean(observed_folds))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        null[i] = float(np.mean(folds.fold_scores(rng.permutation(y), classes)))
    hits = int(np.sum(null >= observed - 1e-12))
    return {
        "status": "ok",
        "balanced_acc": round(observed, 4),
        "fold_scores": [round(float(v), 4) for v in observed_folds],
        "n_folds": int(len(observed_folds)),
        "null_mean": round(float(null.mean()), 4),
        "null_p95": round(float(np.percentile(null, 95)), 4),
        "p_value": (1 + hits) / (1 + n_perm),
        "n_perm": int(n_perm),
        "chance": round(1.0 / len(classes), 4),
        "n_classes": int(len(classes)),
    }


def paired_fold_difference(a: Sequence[float], b: Sequence[float], *,
                           n_boot: int = 10_000, seed: int = 0) -> dict:
    """Paired difference ``a - b`` across matched folds, with a bootstrap CI.

    Both feature sets are scored on the *same* folds, so the fold is the natural
    pairing unit. A CI excluding zero is the evidence that one feature set
    decodes better than the other; comparing two independent intervals, as the
    earlier probe did, is a weaker and less appropriate test.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        raise ValueError(f"need matched folds, got {len(a)} and {len(b)}")
    d = a - b
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "mean_diff": round(float(d.mean()), 4),
        "ci_lo": round(float(lo), 4),
        "ci_hi": round(float(hi), 4),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_folds": int(len(d)),
    }


def holm(pvalues: Mapping[str, float] | Iterable[float]):
    """Holm-Bonferroni adjusted p-values, preserving the input container type.

    Adjusted values are made monotone non-decreasing in rank order and clipped
    at 1, so comparing them against the nominal alpha controls the family-wise
    error rate.
    """
    is_mapping = hasattr(pvalues, "items")
    keys = list(pvalues.keys()) if is_mapping else None
    raw = np.asarray(list(pvalues.values()) if is_mapping else list(pvalues), dtype=float)
    m = len(raw)
    if m == 0:
        return {} if is_mapping else []
    order = np.argsort(raw)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * raw[idx])
        adjusted[idx] = min(running, 1.0)
    if is_mapping:
        return {k: float(v) for k, v in zip(keys, adjusted)}
    return [float(v) for v in adjusted]
