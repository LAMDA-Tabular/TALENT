"""High-level evaluation orchestrator used by both ``TALENT.api.run()`` and
the CLI scripts ``train_model_deep.py`` / ``train_model_classical.py``.

The orchestrator does three things on top of the raw ``method.predict()``
call:

1. **predict_proba standardization** -- converts the method's native
   output (logits / probabilities / class labels) to a uniform ``(N, K)``
   probability array using the ``output_type`` declared in the method
   registry. Methods that return logits get a softmax / sigmoid applied;
   methods that already return probabilities are passed through (with a
   normalize-to-sum-one safety pass).
2. **Threshold tuning on the validation set** -- for binary classification
   only, predicts on the validation split, scans thresholds to maximise a
   chosen metric (default F1), and applies that threshold when computing
   the hard-prediction metrics (Accuracy / F1 / Precision / Recall /
   balanced accuracy). Threshold-independent metrics (AUC, LogLoss, ECE,
   Brier) are unaffected.
3. **Metric recomputation** -- calls ``method.metric()`` again with the
   tuned threshold so the returned tuple already reflects it. The result
   is the single source of truth; downstream code does not need to know
   about the threshold to interpret the metrics.

Both ``threshold`` and ``predict_proba`` flow back to the caller so they
can be inspected, persisted, or reused.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------------
#  Helpers
# ----------------------------------------------------------------------------

def _val_as_test(train_val_data: Tuple[Any, Any, Any]) -> Tuple[Any, Any, Any]:
    """Remap the validation split to look like a ``test`` split for
    :meth:`Method.predict`. ``method.predict`` keys into ``['test']``
    internally; we just relabel the dict.
    """
    N, C, y = train_val_data
    N_test = {"test": N["val"]} if N is not None else None
    C_test = {"test": C["val"]} if C is not None else None
    y_test = {"test": y["val"]}
    return (N_test, C_test, y_test)


def _to_numpy(arr: Any) -> np.ndarray:
    """Convert torch tensors / lists / numpy to a numpy array."""
    if arr is None:
        return None
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr)


def standardize_predict_proba(
    predictions: Any,
    output_type,
    is_regression: bool,
) -> Optional[np.ndarray]:
    """Coerce a method's raw prediction output to ``(N, K)`` probabilities.

    Returns ``None`` for regression or for methods whose ``output_type``
    is ``CLASS_LABELS`` (i.e. nothing to convert).

    The conversion rules are:

    * ``OutputType.PROBABILITIES``: already probabilities; we only re-normalize
      rows in case of numerical drift.
    * ``OutputType.LOGITS``: softmax (sigmoid for 1D binary logits).
    * ``OutputType.CLASS_LABELS``: returns ``None``.
    """
    from TALENT.model.method_registry import OutputType

    if is_regression or output_type == OutputType.CLASS_LABELS:
        return None

    arr = _to_numpy(predictions).astype(np.float64, copy=False)

    if output_type == OutputType.PROBABILITIES:
        if arr.ndim == 1:
            # Positive-class probability for a binary task -> expand.
            return np.stack([1.0 - arr, arr], axis=1)
        # Numerical-drift safety: re-normalize so each row sums to 1.
        s = arr.sum(axis=1, keepdims=True)
        s = np.where(s == 0.0, 1.0, s)
        return arr / s

    # OutputType.LOGITS
    if arr.ndim == 1:
        # Sigmoid for a single binary logit.
        sig = 1.0 / (1.0 + np.exp(-arr))
        return np.stack([1.0 - sig, sig], axis=1)

    # Softmax (numerically stabilised).
    arr = arr - np.max(arr, axis=1, keepdims=True)
    e = np.exp(arr)
    return e / e.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------
#  Main entry point
# ----------------------------------------------------------------------------

def evaluate(
    method: Any,
    train_val_data: Tuple[Any, Any, Any],
    test_data: Tuple[Any, Any, Any],
    info: Dict[str, Any],
    model_name: str,
    *,
    output_type=None,
    tune_threshold: bool = True,
    threshold_metric: str = "f1",
) -> Dict[str, Any]:
    """Predict + (optionally) tune threshold on val + return enriched metrics.

    Parameters
    ----------
    method : Method
        A fitted ``Method`` instance (deep or classical).
    train_val_data : (N, C, y)
        Training/validation data tuple as returned by ``get_dataset``.
        Each element is either ``None`` or a dict with ``'train'`` and
        ``'val'`` keys.
    test_data : (N, C, y)
        Test data tuple. Each element is either ``None`` or a dict with
        a ``'test'`` key.
    info : dict
        Dataset info (``task_type``, ``n_num_features``, ``n_cat_features``).
    model_name : str
        Checkpoint suffix passed to ``method.predict()`` (e.g. ``"best-val"``).
    output_type : MethodSpec.OutputType, optional
        Output type for this method, used to convert logits to probabilities.
        If ``None``, ``predict_proba`` will not be returned.
    tune_threshold : bool, default True
        If True and the task is binary classification, tune the decision
        threshold on the validation set.
    threshold_metric : {"f1", "accuracy", "precision", "recall", "balanced_accuracy"}
        Metric to optimise when tuning the threshold.

    Returns
    -------
    result : dict
        Keys: ``loss``, ``metrics``, ``metric_names``, ``predictions``,
        ``predict_proba``, ``predict_labels``, ``threshold``.

        - ``loss`` -- raw loss as returned by ``method.predict()``
        - ``metrics`` / ``metric_names`` -- tuned-threshold metrics if
          binclass + tune_threshold, otherwise the original metrics
        - ``predictions`` -- raw method output (whatever ``predict()`` returned)
        - ``predict_proba`` -- ``(N, K)`` ndarray of probabilities, or
          ``None`` for regression / class-label methods
        - ``predict_labels`` -- hard predictions consistent with the tuned
          threshold (binclass) or ``argmax(predict_proba)`` otherwise
        - ``threshold`` -- the tuned threshold, or ``None`` if not tuned
    """
    task = info.get("task_type")
    is_regression = task == "regression"
    is_binclass = task == "binclass"

    # ------ 1. Predict on TEST ------------------------------------------------
    test_result = method.predict(test_data, info, model_name=model_name)
    if len(test_result) == 4:
        vl, vres, metric_names, test_predictions = test_result
    elif len(test_result) == 3:
        vres, metric_names, test_predictions = test_result
        vl = float("nan")
    else:
        raise RuntimeError(
            f"Unexpected predict() return arity {len(test_result)} for "
            f"{type(method).__name__!r}"
        )

    # Snapshot the test-time state -- a second predict() call below would
    # overwrite ``method.y_test``.
    test_labels = method.y_test if hasattr(method, "y_test") else None
    y_info = method.y_info if hasattr(method, "y_info") else None

    # ------ 2. Tune threshold on the validation set ---------------------------
    threshold: Optional[float] = None

    if is_binclass and tune_threshold:
        try:
            val_data = _val_as_test(train_val_data)
            val_result = method.predict(val_data, info, model_name=model_name)
            if len(val_result) == 4:
                _, _, _, val_predictions = val_result
            else:
                _, _, val_predictions = val_result
            val_labels = method.y_test if hasattr(method, "y_test") else None

            val_proba = standardize_predict_proba(
                val_predictions, output_type, is_regression=False
            )
            if val_proba is not None and val_proba.shape[1] == 2 and val_labels is not None:
                from TALENT.model.lib.threshold import tune_threshold as _find_threshold
                threshold, _ = _find_threshold(
                    val_labels, val_proba, metric=threshold_metric
                )

                # Recompute the test-set metrics using the tuned threshold.
                vres, metric_names = method.metric(
                    test_predictions, test_labels, y_info, threshold=threshold
                )
        except Exception as exc:  # pragma: no cover -- defensive fallback
            warnings.warn(
                f"Threshold tuning failed ({exc!r}); "
                f"falling back to argmax metrics."
            )
            threshold = None

    # ------ 3. Build predict_proba / predict_labels --------------------------
    test_proba = standardize_predict_proba(
        test_predictions, output_type, is_regression=is_regression
    )
    predict_labels: Optional[np.ndarray] = None
    if test_proba is not None:
        if is_binclass and threshold is not None:
            predict_labels = (test_proba[:, 1] >= threshold).astype(int)
        else:
            predict_labels = test_proba.argmax(axis=1)

    return {
        "loss": float(vl) if vl == vl else float("nan"),  # NaN-safe coercion
        "metrics": tuple(vres),
        "metric_names": tuple(metric_names),
        "predictions": test_predictions,
        "predict_proba": test_proba,
        "predict_labels": predict_labels,
        "threshold": threshold,
    }
