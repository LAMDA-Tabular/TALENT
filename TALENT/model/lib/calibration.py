"""Calibration metrics for probabilistic classifiers.

Both metrics here are reported as **lower-is-better**. Perfect calibration
corresponds to a score of ``0.0``.

* :func:`brier_score` — mean squared error between predicted probabilities
  and one-hot encoded ground truth. For binary classification this reduces
  to ``mean((p_pos - y)^2)``.
* :func:`expected_calibration_error` — equal-width bin estimate of
  :math:`E[|P(correct | confidence) - confidence|]`, the standard ECE
  formulation used in (Guo et al., 2017).
"""

from __future__ import annotations

import numpy as np


def _ensure_proba_array(predictions: np.ndarray) -> np.ndarray:
    """Coerce ``predictions`` into an ``(N, K)`` probability array.

    Accepts:
      - 2D array of shape ``(N, K)`` -- returned as-is
      - 1D array of shape ``(N,)`` -- interpreted as positive-class
        probability of a binary task, expanded to ``(N, 2)``
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim == 1:
        predictions = np.stack([1.0 - predictions, predictions], axis=1)
    return predictions


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Multi-class Brier score (lower is better; perfect = 0).

    For binary classification with ``y_proba`` shape ``(N, 2)``, this is
    equivalent to ``mean((y_proba[:, 1] - y_true) ** 2)``. For ``K``
    classes the multi-class generalisation is
    ``mean(sum((y_proba - one_hot(y_true)) ** 2, axis=1))``.

    Parameters
    ----------
    y_true : (N,) ndarray
        Ground-truth class indices (``0..K-1``).
    y_proba : (N, K) ndarray
        Predicted class probabilities. Each row should sum to ~1.0.
    """
    y_proba = _ensure_proba_array(y_proba)
    y_true = np.asarray(y_true).astype(int).ravel()
    n_classes = y_proba.shape[1]

    if n_classes == 2:
        # Faster path that matches the standard binary Brier definition.
        return float(np.mean((y_proba[:, 1] - y_true) ** 2))

    # Multi-class: sum of squared differences over the one-hot encoding.
    one_hot = np.zeros_like(y_proba)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (Guo et al., 2017; lower is better).

    Buckets samples by predicted confidence (max class probability) into
    ``n_bins`` equal-width bins on ``[0, 1]``, and reports the weighted
    average gap between confidence and accuracy inside each bin.

    Parameters
    ----------
    y_true : (N,) ndarray
        Ground-truth class indices.
    y_proba : (N, K) ndarray
        Predicted class probabilities.
    n_bins : int, default 15
        Number of equal-width bins.
    """
    y_proba = _ensure_proba_array(y_proba)
    y_true = np.asarray(y_true).astype(int).ravel()

    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Closed upper edge on the last bin so confidence==1.0 is counted.
        if i == n_bins - 1:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)
        n_in = int(in_bin.sum())
        if n_in > 0:
            avg_conf = float(confidences[in_bin].mean())
            avg_acc = float(accuracies[in_bin].mean())
            ece += (n_in / n) * abs(avg_conf - avg_acc)

    return float(ece)
