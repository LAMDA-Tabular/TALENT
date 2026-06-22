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


def validate_tune_metric(name: Optional[str]) -> Optional[str]:
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

    Falls back to the historical behaviour --- the first metric in ``vres``,
    with ``higher_is_better == (not is_regression)`` --- when no ``tune_metric``
    is configured, when the requested metric is not available for this task
    (e.g. ``AUC`` on a regression task), or when the value is NaN (e.g. ``AUC``
    on a degenerate single-class validation fold).
    """
    legacy = (float(vres[0]), (not is_regression))
    key = resolve_tune_metric(args)
    if not key or key not in metric_names:
        return legacy
    val = vres[list(metric_names).index(key)]
    try:
        val = float(val)
    except (TypeError, ValueError):
        return legacy
    if math.isnan(val):
        return legacy
    return val, _HIGHER_IS_BETTER.get(key, not is_regression)


def study_direction(args, is_regression: bool) -> str:
    """Optuna study direction (``'maximize'``/``'minimize'``) for the metric.

    Mirrors :func:`select_objective`: defaults to ``'minimize'`` for regression
    and ``'maximize'`` for classification when no ``tune_metric`` is set.
    """
    key = resolve_tune_metric(args)
    if not key or key not in _HIGHER_IS_BETTER:
        return "minimize" if is_regression else "maximize"
    return "maximize" if _HIGHER_IS_BETTER[key] else "minimize"


def worst_objective_value(args, is_regression: bool) -> float:
    """A sentinel "worst" score for the configured direction.

    Used as the objective's return value when a trial raises, so a failed
    trial is never selected as best regardless of optimization direction.
    """
    return -1e18 if study_direction(args, is_regression) == "maximize" else 1e18


# Methods whose HPO objective is an internal training loss rather than a value
# returned by ``Method.metric`` (so a user-selected ``tune_metric`` cannot be
# applied to them). Tuning these still works with ``tune_metric=None``.
METHODS_WITHOUT_TUNE_METRIC = frozenset({"tabnet", "ptarl", "tabcaps"})


def assert_tune_metric_supported(model_type: str, args) -> None:
    """Raise if a ``tune_metric`` is set for a method that cannot honour it."""
    if resolve_tune_metric(args) is not None and model_type in METHODS_WITHOUT_TUNE_METRIC:
        raise NotImplementedError(
            f"tune_metric is not supported for method {model_type!r}, which tunes "
            f"on its internal training loss rather than a Method.metric output. "
            f"Use tune_metric=None for this method, or exclude it from "
            f"metric-specific HPO."
        )
