"""Threshold tuning for binary classification.

Many real-world deployments tune the decision threshold on a held-out
validation set rather than using the default ``argmax`` (i.e. ``0.5`` for
binary classification). This is especially important for imbalanced
datasets where the F1-optimal threshold is often far from ``0.5``.

This module provides a single helper, :func:`tune_threshold`, that scans
a dense grid (with extra resolution near 0 and 1 for highly imbalanced
data) and returns the threshold that maximises a chosen metric.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np


def _positive_class_proba(y_proba: np.ndarray) -> np.ndarray:
    """Coerce ``y_proba`` to a 1D positive-class probability array."""
    y_proba = np.asarray(y_proba, dtype=np.float64)
    if y_proba.ndim == 1:
        return y_proba
    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        return y_proba[:, 1]
    raise ValueError(
        f"tune_threshold expects binary probabilities, got shape {y_proba.shape}."
    )


def tune_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
    n_thresholds: int = 101,
) -> Tuple[float, float]:
    """Find the decision threshold that maximises ``metric`` on ``(y_true, y_proba)``.

    Parameters
    ----------
    y_true : (N,) ndarray
        Ground-truth binary labels in ``{0, 1}``.
    y_proba : (N,) or (N, 2) ndarray
        Predicted positive-class probability (or full ``(N, 2)`` proba
        array, in which case column 1 is used).
    metric : {"f1", "accuracy", "precision", "recall", "balanced_accuracy"}
        Metric to optimise. Defaults to ``"f1"`` because it is meaningful
        on imbalanced data and is the most common choice in tabular work.
    n_thresholds : int, default 101
        Number of thresholds in the mid-range ``[0.01, 0.99]`` grid. The
        helper additionally adds 20 finely-spaced thresholds in each tail
        ``[1e-4, 1e-2]`` and ``[0.99, 1 - 1e-4]`` to handle imbalanced
        problems where the optimum is close to 0 or 1.

    Returns
    -------
    (threshold, score) : Tuple[float, float]
        Optimal threshold (``proba >= threshold`` predicts class 1) and
        the achieved score for that threshold.
    """
    from sklearn.metrics import (
        f1_score,
        accuracy_score,
        precision_score,
        recall_score,
        balanced_accuracy_score,
    )

    pos_proba = _positive_class_proba(y_proba)
    y_true = np.asarray(y_true).astype(int).ravel()

    metric_fns = {
        "f1": lambda y_t, y_p: f1_score(y_t, y_p, zero_division=0),
        "accuracy": accuracy_score,
        "precision": lambda y_t, y_p: precision_score(y_t, y_p, zero_division=0),
        "recall": lambda y_t, y_p: recall_score(y_t, y_p, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score,
    }
    if metric not in metric_fns:
        raise ValueError(
            f"Unknown metric {metric!r}. Choose one of {sorted(metric_fns)}."
        )
    metric_fn = metric_fns[metric]

    # Dense grid in the middle, fine-grained tails for highly imbalanced data.
    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(1e-4, 1e-2, 20),
                np.linspace(0.01, 0.99, n_thresholds),
                np.linspace(0.99, 1.0 - 1e-4, 20),
            ]
        )
    )

    best_score = -np.inf
    best_t = 0.5
    for t in thresholds:
        y_pred = (pos_proba >= t).astype(int)
        score = float(metric_fn(y_true, y_pred))
        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, float(best_score)
