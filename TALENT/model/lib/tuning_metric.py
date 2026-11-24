"""Selectable objective metric for hyper-parameter optimization (HPO).

By default TALENT optimizes the *first* metric returned by ``Method.metric``
during tuning (historically: Accuracy for classification, MAE for the deep
regressors / RMSE for the classical regressors). Setting ``args.tune_metric``
to any metric name that ``Method.metric`` can emit makes the HPO search
optimize that metric instead, in the correct direction.

This is a generic TALENT feature: ``tune_metric`` is ``None`` by default, in
which case the historical behaviour is preserved exactly, so existing results
and downstream code are unaffected.

Supported metric names (must match the names returned by ``Method.metric``):

* Classification: ``Accuracy``, ``Avg_Recall``, ``Avg_Precision``, ``F1``,
  ``LogLoss``, ``AUC``, ``Brier``, ``ECE``.
* Regression: ``MAE``, ``R2``, ``RMSE``.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

# Whether a larger value of each metric means a better model. This is the
# single source of truth for the optimization direction during HPO.
_HIGHER_IS_BETTER = {
    # --- classification ---
    "Accuracy": True,
    "Avg_Recall": True,
    "Avg_Precision": True,
    "F1": True,
    "AUC": True,
    "LogLoss": False,
    "Brier": False,
    "ECE": False,
    # --- regression ---
    "R2": True,
    "MAE": False,
    "RMSE": False,
}


def supported_tune_metrics() -> Tuple[str, ...]:
    """Every metric name that can be passed as ``tune_metric``."""
    return tuple(_HIGHER_IS_BETTER.keys())


def metric_higher_is_better(name: str) -> bool:
    """Return True if a larger value of ``name`` is better."""
    return _HIGHER_IS_BETTER[name]


def validate_tune_metric(name: Optional[str], *, is_regression: Optional[bool] = None) -> Optional[str]:
    """Validate a requested ``tune_metric``; return it unchanged (or ``None``).

    Raises ``ValueError`` for an unknown metric name so misconfiguration fails
    fast rather than silently falling back.
    """
    if name is None:
        return None
    if name not in _HIGHER_IS_BETTER:
        raise ValueError(
            f"Unknown tune_metric {name!r}. Supported metrics: "
            f"{', '.join(supported_tune_metrics())}."
        )
    if is_regression is not None and (name in {"MAE", "RMSE", "R2"}) != is_regression:
        task = "regression" if is_regression else "classification"
        raise ValueError(f"HPO metric {name} is unavailable for {task}")
    return name


def resolve_tune_metric(args) -> Optional[str]:
    """Read ``args.tune_metric`` defensively (``None`` if unset)."""
    return getattr(args, "tune_metric", None)


def select_objective(
    vres: Sequence[float],
    metric_names: Sequence[str],
    args,
    is_regression: bool,
) -> Tuple[float, bool]:
    """Pick the HPO objective value + direction from a ``Method.metric`` result.

    :param vres: the metric values returned by ``Method.metric``.
    :param metric_names: the matching metric names.
    :param args: the run arguments (read ``args.tune_metric``).
    :param is_regression: whether the task is regression.
    :return: ``(score, higher_is_better)``.

    The first metric supplies the historical default. An explicitly selected
    metric must be available and finite on the validation data.
    """
    key = validate_tune_metric(resolve_tune_metric(args), is_regression=is_regression)
    if key is None:
        return float(vres[0]), not is_regression
    if key not in metric_names:
        raise ValueError(f"The method did not report the requested HPO metric {key}")
    score = float(vres[metric_names.index(key)])
    if not math.isfinite(score):
        raise ValueError(f"HPO metric {key} is undefined on this validation split")
    return score, _HIGHER_IS_BETTER[key]


def study_direction(args, is_regression: bool) -> str:
    """Optuna study direction (``'maximize'``/``'minimize'``) for the metric.

    Mirrors :func:`select_objective`: defaults to ``'minimize'`` for regression
    and ``'maximize'`` for classification when no ``tune_metric`` is set.
    """
    key = validate_tune_metric(resolve_tune_metric(args), is_regression=is_regression)
    if key is None:
        return "minimize" if is_regression else "maximize"
    return "maximize" if _HIGHER_IS_BETTER[key] else "minimize"


def worst_objective_value(args, is_regression: bool) -> float:
    """A sentinel "worst" score for the configured direction.

    Retained for callers using numeric failure scores. TALENT's Optuna
    studies record unsuccessful trials with the failed state.
    """
    return -1e18 if study_direction(args, is_regression) == "maximize" else 1e18


# These adapters use a fixed validation metric for single-validation HPO.
METHODS_WITHOUT_TUNE_METRIC = frozenset({"tabnet", "ptarl", "tabcaps"})


def assert_tune_metric_supported(model_type: str, args) -> None:
    """Raise if a ``tune_metric`` is set for a method that cannot honour it."""
    if resolve_tune_metric(args) is not None and model_type in METHODS_WITHOUT_TUNE_METRIC:
        raise NotImplementedError(
            f"tune_metric is not supported for single-validation HPO with {model_type!r}. "
            "Use tune_metric=None for the adapter's validation metric, or select "
            "a scoring metric with cross_validate(..., tune=True, hpo_cv=...)."
        )
