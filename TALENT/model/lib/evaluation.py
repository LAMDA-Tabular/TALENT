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

from TALENT.benchmark.measurement import elapsed_call


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

    if is_regression or output_type in (None, OutputType.CLASS_LABELS):
        return None

    arr = _to_numpy(predictions).astype(np.float64, copy=False)

    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]

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
        sig = np.exp(-np.logaddexp(0.0, -arr))
        return np.stack([1.0 - sig, sig], axis=1)

    # Softmax (numerically stabilised).
    arr = arr - np.max(arr, axis=1, keepdims=True)
    e = np.exp(arr)
    return e / e.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------
#  Main entry point
# ----------------------------------------------------------------------------

def _custom_score(scorer, train_labels, test_labels, predictions, probabilities, labels):
    from sklearn.preprocessing import LabelEncoder

    targets = np.asarray(test_labels).reshape(-1)
    if scorer.response == "prediction":
        response = _to_numpy(predictions).reshape(-1)
    else:
        targets = LabelEncoder().fit(np.asarray(train_labels).reshape(-1)).transform(targets)
        response = probabilities if scorer.response == "proba" else labels
        if response is None:
            if scorer.response == "proba":
                raise ValueError(f"Scorer {scorer.name!r} requires probabilities")
            response = _to_numpy(predictions).reshape(-1)
    return scorer(targets, response)


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
    decision_threshold: Optional[float] = None,
    return_validation_predictions: bool = False,
    scoring=None,
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
    decision_threshold : float, optional
        Apply this previously selected binary threshold directly.

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
    from TALENT.benchmark.scoring import Scorer

    task = info.get("task_type")
    is_regression = task == "regression"
    is_binclass = task == "binclass"

    # ------ 1. Predict on TEST ------------------------------------------------
    device = getattr(getattr(method, "args", None), "device", None)
    test_result, predict_time = elapsed_call(
        "predict_seconds", device, method.predict, test_data, info, model_name=model_name
    )
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

    test_labels = method.y_test if hasattr(method, "y_test") else None
    y_info = method.y_info if hasattr(method, "y_info") else None
    test_proba = standardize_predict_proba(test_predictions, output_type, is_regression)

    # ------ 2. Tune threshold on the validation set ---------------------------
    threshold: Optional[float] = None
    validation_predictions = validation_proba = None
    select_threshold = decision_threshold is None and is_binclass and tune_threshold

    if decision_threshold is not None:
        threshold = decision_threshold
        scores = test_predictions if test_proba is None else test_proba
        vres, metric_names = method.metric(scores, test_labels, y_info, threshold=threshold)
    if select_threshold or return_validation_predictions:
        # Adapters replace their query arrays and loader on each predict call.
        test_state = {name: getattr(method, name)
                      for name in ("N_test", "C_test", "y_test", "test_loader", "predict_time")
                      if hasattr(method, name)}
        try:
            val_data = _val_as_test(train_val_data)
            val_result = method.predict(val_data, info, model_name=model_name)
            if len(val_result) == 4:
                _, _, _, val_predictions = val_result
            else:
                _, _, val_predictions = val_result
            val_labels = method.y_test if hasattr(method, "y_test") else None

            val_proba = standardize_predict_proba(
                val_predictions, output_type, is_regression=is_regression
            )
            if return_validation_predictions:
                validation_predictions = _to_numpy(val_predictions)
                validation_proba = val_proba
            if select_threshold and val_proba is not None and val_proba.shape[1] == 2 and val_labels is not None:
                from TALENT.model.lib.threshold import tune_threshold as _find_threshold
                threshold, _ = _find_threshold(
                    _to_numpy(val_labels), val_proba, metric=threshold_metric
                )

                # Recompute the test-set metrics using the tuned threshold.
                vres, metric_names = method.metric(
                    test_proba, test_labels, y_info, threshold=threshold
                )
        except Exception as exc:  # pragma: no cover -- defensive fallback
            if return_validation_predictions or isinstance(threshold_metric, Scorer):
                raise
            warnings.warn(
                f"Threshold tuning failed ({exc!r}); "
                f"falling back to argmax metrics."
            )
            threshold = None
        finally:
            for name, value in test_state.items():
                setattr(method, name, value)

    # ------ 3. Build predict_proba / predict_labels --------------------------
    predict_labels: Optional[np.ndarray] = None
    if test_proba is not None:
        if is_binclass and threshold is not None:
            predict_labels = (test_proba[:, 1] >= threshold).astype(int)
        else:
            predict_labels = test_proba.argmax(axis=1)

    if scoring is not None:
        if scoring.name in metric_names:
            raise ValueError(f"Custom scorer name {scoring.name!r} duplicates a reported metric")
        value = _custom_score(scoring, train_val_data[2]["train"], test_data[2]["test"],
                              test_predictions, test_proba, predict_labels)
        vres, metric_names = (*vres, value), (*metric_names, scoring.name)

    return {
        "validation_predictions": validation_predictions,
        "validation_predict_proba": validation_proba,
        "predict_time": predict_time,
        "loss": float(vl) if vl == vl else float("nan"),  # NaN-safe coercion
        "metrics": tuple(vres),
        "metric_names": tuple(metric_names),
        "predictions": test_predictions,
        "predict_proba": test_proba,
        "predict_labels": predict_labels,
        "threshold": threshold,
    }
