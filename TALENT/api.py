"""Public Python API for TALENT.

TALENT was originally designed as a CLI tool with `train_model_deep.py` /
`train_model_classical.py` driving everything through argparse on
`sys.argv`. This module exposes the same functionality as a regular Python
API, so downstream code (notebooks, benchmarking pipelines, scikit-learn
wrappers) can call into TALENT without manipulating argv.

Quick examples
--------------

>>> import TALENT
>>> from TALENT.model.lib.data import get_dataset
>>>
>>> train_val, test, info = get_dataset("housing", "./data")
>>> result = TALENT.run("catboost", train_val, test, info)
>>> result.metrics, result.metric_names
>>> result.predictions  # in original target scale for regression

>>> # Hyperparameter tuning:
>>> result = TALENT.run("mlp", train_val, test, info, tune=True, n_trials=50)

>>> # Multiple seeds, aggregated:
>>> result = TALENT.run("tabpfn_v3", train_val, test, info, seed_num=3)
>>> result.metrics_mean, result.metrics_std

>>> # Introspect a method:
>>> spec = TALENT.get_method_spec("tabpfn_v2")
>>> spec.cat_policy, spec.normalization, spec.supports_hpo

>>> # List methods by capability:
>>> from TALENT.model.method_registry import Architecture
>>> [s.name for s in TALENT.list_methods(architecture=Architecture.DEEP)]

The CLI scripts continue to work unchanged. Both surfaces go through the
same `MethodSpec` registry, so adding a new method requires no changes to
either layer beyond a registry entry.
"""

from __future__ import annotations

import os
import os.path as osp
import json
import time
import tempfile
import warnings
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import importlib.resources as pkg_resources
import numpy as np

from TALENT.model.method_registry import (
    METHOD_REGISTRY,
    MethodSpec,
    Architecture,
    Hardware,
    OutputType,
    get_method_spec,
    get_method_class,
    list_methods,
    deep_method_names,
    classical_method_names,
    all_method_names,
)


# ----------------------------------------------------------------------------
#  Result dataclass
# ----------------------------------------------------------------------------

@dataclass
class RunResult:
    """Output of a single :func:`run` call.

    For single-seed runs, the ``*_mean`` / ``*_std`` fields equal the
    single value and ``0.0`` respectively, so the API shape is stable.

    Notes on the prediction-related fields:

    - ``predictions`` is the method's *raw* output (logits for some
      methods, probabilities for others, regression scalars for regressors).
      Regression predictions are returned on the **original target scale**
      (i.e. denormalized).
    - ``predict_proba`` is a uniform ``(N, K)`` probability array for
      every classifier regardless of whether its native output is logits
      or probabilities. ``None`` for regression and class-label-only
      methods.
    - ``predict_labels`` is the corresponding hard prediction. For binary
      classification with threshold tuning, it uses the tuned threshold;
      otherwise it uses ``argmax(predict_proba)``.
    - ``threshold`` is the decision threshold tuned on the validation set
      (binary classification only). ``None`` otherwise.
    """
    predictions: np.ndarray
    loss: float
    metrics: Tuple[float, ...]
    metric_names: Tuple[str, ...]
    fit_time: float
    predict_time: float

    # Multi-seed aggregates (filled when seed_num > 1)
    loss_mean: float = 0.0
    loss_std: float = 0.0
    metrics_mean: Tuple[float, ...] = field(default_factory=tuple)
    metrics_std: Tuple[float, ...] = field(default_factory=tuple)
    per_seed: List[Dict[str, Any]] = field(default_factory=list)

    # Standardized probabilistic output + tuned threshold (last seed)
    predict_proba: Optional[np.ndarray] = None
    predict_labels: Optional[np.ndarray] = None
    threshold: Optional[float] = None

    # Bookkeeping
    method_type: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    save_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe summary (predictions excluded)."""
        d = {
            "method_type": self.method_type,
            "loss": float(self.loss),
            "loss_mean": float(self.loss_mean),
            "loss_std": float(self.loss_std),
            "metrics": dict(zip(self.metric_names, [float(x) for x in self.metrics])),
            "metrics_mean": dict(zip(self.metric_names, [float(x) for x in self.metrics_mean])) if self.metrics_mean else {},
            "metrics_std": dict(zip(self.metric_names, [float(x) for x in self.metrics_std])) if self.metrics_std else {},
            "fit_time": float(self.fit_time),
            "predict_time": float(self.predict_time),
            "threshold": float(self.threshold) if self.threshold is not None else None,
            "save_path": self.save_path,
        }
        return d


# ----------------------------------------------------------------------------
#  Args builder (replaces argparse + sys.argv for programmatic use)
# ----------------------------------------------------------------------------

# Default values that mirror configs/deep_configs.json and classical_configs.json
# but are baked in so the API works even when no JSON file is found.
_DEEP_DEFAULTS = {
    "dataset": "",
    "max_epoch": 200,
    "batch_size": 1024,
    "normalization": "standard",
    "num_nan_policy": "mean",
    "cat_nan_policy": "new",
    "cat_policy": "indices",
    "num_policy": "none",
    "n_bins": 2,
    "cat_min_frequency": 0.0,
    "n_trials": 50,
    "seed_num": 1,
    "workers": 0,
    "gpu": "0",
    "tune": False,
    "retune": False,
    "evaluate_option": "best-val",
    "dataset_path": "./data",
    "model_path": "./checkpoints",
    "use_float": False,
}

_CLASSICAL_DEFAULTS = {
    "dataset": "",
    "normalization": "standard",
    "num_nan_policy": "mean",
    "cat_nan_policy": "new",
    "cat_policy": "ohe",
    "num_policy": "none",
    "n_bins": 2,
    "cat_min_frequency": 0.0,
    "n_trials": 50,
    "seed_num": 1,
    "gpu": "0",
    "tune": False,
    "retune": False,
    "dataset_path": "./data",
    "model_path": "./checkpoints",
    "evaluate_option": "best-val",
    "use_float": False,
}


def _try_load_packaged_defaults(filename: str) -> Dict[str, Any]:
    """Best-effort: load packaged `configs/<filename>` defaults if present."""
    try:
        import TALENT
        with pkg_resources.files(TALENT).joinpath(f"configs/{filename}").open("r") as f:
            return json.load(f)
    except Exception:
        return {}


def _try_load_method_config(model_type: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (default_config, opt_space_config) for `model_type` from
    packaged JSONs. Returns empty dicts if not available.
    """
    default_cfg: Dict[str, Any] = {}
    opt_cfg: Dict[str, Any] = {}
    try:
        import TALENT
        with pkg_resources.files(TALENT).joinpath(f"configs/default/{model_type}.json").open("r") as f:
            default_cfg = json.load(f)
    except Exception:
        pass
    try:
        import TALENT
        with pkg_resources.files(TALENT).joinpath(f"configs/opt_space/{model_type}.json").open("r") as f:
            opt_cfg = json.load(f)
    except Exception:
        pass
    return default_cfg, opt_cfg


def build_args(
    model_type: str,
    *,
    save_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    apply_spec_constraints: bool = True,
    **overrides,
) -> SimpleNamespace:
    """Build an `args` namespace equivalent to what argparse would produce.

    Parameters
    ----------
    model_type : str
        Name from the method registry (e.g. ``"tabpfn_v2"``, ``"catboost"``).
    save_path : str, optional
        Working directory for checkpoints / tuned configs. If None, a fresh
        temp directory is created.
    config : dict, optional
        Method config (``{"model": {...}, "training": {...}, "general": {...}}``).
        If None, the packaged default config is loaded.
    apply_spec_constraints : bool, default True
        If True, override `cat_policy`, `normalization`, and `num_policy`
        with the values required by the method spec (so the asserts in
        each method's `__init__` cannot fail).
    **overrides
        Anything else (``seed``, ``batch_size``, ``max_epoch``, ``tune``,
        ``n_trials``, ...). These override the defaults and any packaged
        config values.

    Returns
    -------
    SimpleNamespace
        Drop-in replacement for `argparse.Namespace`, accepted by every
        Method class.
    """
    spec = get_method_spec(model_type)

    # Start from architecture-appropriate defaults
    base = (_DEEP_DEFAULTS if spec.architecture == Architecture.DEEP
            else _CLASSICAL_DEFAULTS).copy()

    # Layer in packaged config defaults if they exist
    packaged = _try_load_packaged_defaults(
        "deep_configs.json" if spec.architecture == Architecture.DEEP
        else "classical_configs.json"
    )
    for k, v in packaged.items():
        if k in base:  # only override known fields, ignore extras
            base[k] = v

    # User overrides win
    base["model_type"] = model_type
    for k, v in overrides.items():
        base[k] = v

    # Enforce method-spec preprocessing constraints
    if apply_spec_constraints:
        if spec.cat_policy is not None and base.get("cat_policy") not in spec.cat_policy:
            base["cat_policy"] = spec.cat_policy[0]
        if spec.normalization is not None:
            base["normalization"] = spec.normalization
        if spec.num_policy is not None:
            base["num_policy"] = spec.num_policy

    # Resolve save_path
    if save_path is None:
        save_path = tempfile.mkdtemp(prefix=f"talent-{model_type}-")
    base["save_path"] = save_path
    os.makedirs(save_path, exist_ok=True)

    # Default seed
    base.setdefault("seed", 0)

    # Method config: packaged default, then user override
    if config is None:
        default_cfg, _ = _try_load_method_config(model_type)
        # default_cfg is wrapped one level: { model_type: { model/training/general } }
        if model_type in default_cfg:
            config = default_cfg[model_type]
        else:
            # Minimal fallback so methods that read training/lr etc. still work
            config = {"model": {}, "training": {}, "general": {}}
    base["config"] = config

    # n_bins propagation matches the CLI behaviour
    try:
        if spec.architecture == Architecture.DEEP:
            base["config"].setdefault("training", {})["n_bins"] = base["n_bins"]
        else:
            base["config"].setdefault("fit", {})["n_bins"] = base["n_bins"]
    except Exception:
        pass

    base.setdefault("tune_metric", None)
    args = SimpleNamespace(**base)

    # Validate so we fail with a readable message rather than an obscure assert.
    if apply_spec_constraints:
        spec.validate_args(args)

    return args


# ----------------------------------------------------------------------------
#  Core run function
# ----------------------------------------------------------------------------

def _aggregate_metrics(per_seed: List[Dict[str, Any]], metric_names: Tuple[str, ...]):
    losses = np.array([s["loss"] for s in per_seed], dtype=float)
    metrics_matrix = np.array([list(s["metrics"]) for s in per_seed], dtype=float)
    return (
        float(losses.mean()),
        float(losses.std()),
        tuple(metrics_matrix.mean(axis=0).tolist()),
        tuple(metrics_matrix.std(axis=0).tolist()),
    )


def run(
    model_type: str,
    train_val_data: Tuple[Any, Any, Any],
    test_data: Tuple[Any, Any, Any],
    info: Dict[str, Any],
    *,
    config: Optional[Dict[str, Any]] = None,
    args: Optional[SimpleNamespace] = None,
    seed: int = 0,
    seed_num: int = 1,
    tune: bool = False,
    n_trials: int = 50,
    save_path: Optional[str] = None,
    return_method: bool = False,
    verbose: bool = False,
    tune_threshold: bool = True,
    threshold_metric: str = "f1",
    tune_metric: Optional[str] = None,
    **arg_overrides,
) -> RunResult:
    """Fit a TALENT method on ``train_val_data`` and predict on ``test_data``.

    Parameters
    ----------
    model_type : str
        Name from the registry (``TALENT.list_methods()`` to enumerate).
    train_val_data : (N, C, y)
        Tuple of dicts keyed by 'train' / 'val', as returned by
        :func:`TALENT.model.lib.data.get_dataset`. Each element may be
        ``None`` if the dataset has no numerical / categorical features.
    test_data : (N, C, y)
        Same shape as `train_val_data` but keyed by ``'test'``.
    info : dict
        Dataset info dict (`task_type`, `n_num_features`, `n_cat_features`).
    config : dict, optional
        Method-specific config. If None, the packaged default is used.
    args : SimpleNamespace, optional
        Pre-built args. If supplied, all of the per-call kwargs below are
        ignored (you can also pass `seed` to override `args.seed`). Pass
        the output of :func:`build_args` here when you want fine control.
    seed : int, default 0
        Random seed (used when seed_num == 1).
    seed_num : int, default 1
        Run multiple seeds (0..seed_num-1) and aggregate metrics. The
        returned `predictions` are from the *last* seed; per-seed details
        are in `result.per_seed`.
    tune : bool, default False
        If True and the method supports HPO, run Optuna tuning first.
    n_trials : int, default 50
        Optuna trial budget (when tune=True).
    save_path : str, optional
        Where to save checkpoints / tuned configs. Auto-generated if None.
    return_method : bool, default False
        If True, attach the fitted method object to ``result.method``.
        Adds RAM cost, so off by default.
    verbose : bool, default False
        Forwarded to the method when supported.
    tune_threshold : bool, default True
        For binary classification only: tune the decision threshold on
        the validation split (already required by TALENT) to maximise the
        ``threshold_metric``, and apply that threshold when computing
        Accuracy / F1 / Avg_Recall / Avg_Precision. Threshold-independent
        metrics (LogLoss, AUC, Brier, ECE) are unaffected. Has no effect
        on regression or multiclass tasks.
    threshold_metric : {"f1", "accuracy", "precision", "recall", "balanced_accuracy"}
        Which metric to optimise when tuning the threshold. Default ``"f1"``.
    tune_metric : str, optional
        Metric optimised during hyper-parameter search (e.g. ``"AUC"``,
        ``"R2"``, ``"RMSE"``, ``"LogLoss"``). ``None`` (default) keeps TALENT's
        historical HPO objective (validation Accuracy for classification,
        MAE/RMSE for regression). See
        ``TALENT.model.lib.tuning_metric.supported_tune_metrics()``. Not
        supported for methods that tune on an internal loss (``tabnet``,
        ``ptarl``, ``tabcaps``).
    **arg_overrides
        Passed through to :func:`build_args`.

    Returns
    -------
    RunResult
    """
    spec = get_method_spec(model_type)

    if tune and not spec.supports_hpo:
        raise ValueError(
            f"Method {model_type!r} does not support HPO. "
            f"Set tune=False or pick a tunable method."
        )

    from TALENT.model.lib.tuning_metric import (
        validate_tune_metric, assert_tune_metric_supported,
    )
    validate_tune_metric(tune_metric)

    # Build args if not provided
    if args is None:
        args = build_args(
            model_type,
            save_path=save_path,
            config=config,
            seed=seed,
            seed_num=seed_num,
            tune=tune,
            n_trials=n_trials,
            tune_metric=tune_metric,
            **arg_overrides,
        )
    else:
        # Honour late overrides
        if seed is not None:
            args.seed = seed
        if save_path is not None:
            args.save_path = save_path
            os.makedirs(save_path, exist_ok=True)
        if tune_metric is not None:
            args.tune_metric = tune_metric

    # Lazy imports — we don't want to drag torch in just for `list_methods()`.
    import torch
    from TALENT.model.utils import (
        set_seeds, tune_hyper_parameters,
    )

    # Resolve device
    if not hasattr(args, "device"):
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Detect regression
    is_regression = info.get('task_type') == 'regression'
    if is_regression and not spec.supports_regression:
        raise ValueError(
            f"Method {model_type!r} does not support regression. "
            f"Use a method where supports_regression is True."
        )
    if (not is_regression) and not spec.supports_classification:
        raise ValueError(
            f"Method {model_type!r} does not support classification. "
            f"Use a method where supports_classification is True."
        )

    method_cls = spec.get_class()

    # Optional HPO. The current `tune_hyper_parameters` mutates args.config.
    if tune and spec.supports_hpo:
        assert_tune_metric_supported(model_type, args)
        # Load opt_space — required by tune_hyper_parameters
        _, opt_space = _try_load_method_config(model_type)
        if not opt_space:
            raise FileNotFoundError(
                f"opt_space config for {model_type!r} not found in packaged configs."
            )
        args = tune_hyper_parameters(args, opt_space, train_val_data, info)

    # Use the centralized evaluator so the CLI scripts and the API share
    # the exact same evaluation logic (predict_proba standardization +
    # threshold tuning on val).
    from TALENT.model.lib.evaluation import evaluate

    per_seed: List[Dict[str, Any]] = []
    last_predictions: Optional[np.ndarray] = None
    last_metric_names: Tuple[str, ...] = ()
    last_predict_proba: Optional[np.ndarray] = None
    last_predict_labels: Optional[np.ndarray] = None
    last_threshold: Optional[float] = None
    fit_times: List[float] = []
    predict_times: List[float] = []
    method = None

    for s in range(seed_num):
        args.seed = s if seed_num > 1 else seed
        set_seeds(args.seed)
        if verbose:
            print(f"[TALENT.api.run] seed={args.seed} method={model_type}")

        method = method_cls(args, is_regression)
        fit_ret = method.fit(train_val_data, info)
        # Some methods return time_cost from fit; some set self.fit_time. We use
        # self.fit_time when available (set in our bugfix sweep across methods).
        fit_time = getattr(method, "fit_time", None)
        if fit_time is None and isinstance(fit_ret, (int, float)):
            fit_time = float(fit_ret)
        elif fit_time is None:
            fit_time = 0.0

        eval_result = evaluate(
            method,
            train_val_data,
            test_data,
            info,
            model_name=args.evaluate_option,
            output_type=spec.output_type,
            tune_threshold=tune_threshold,
            threshold_metric=threshold_metric,
        )
        vl = eval_result["loss"]
        vres = eval_result["metrics"]
        metric_names = eval_result["metric_names"]
        preds = eval_result["predictions"]

        # Coerce raw predictions to numpy (predict_proba / labels are already numpy).
        if hasattr(preds, "detach"):
            preds = preds.detach().cpu().numpy()
        preds = np.asarray(preds)

        predict_time = float(getattr(method, "predict_time", 0.0))
        fit_times.append(float(fit_time))
        predict_times.append(predict_time)

        per_seed.append({
            "seed": args.seed,
            "loss": float(vl) if vl == vl else float("nan"),  # NaN-safe
            "metrics": tuple(float(x) for x in vres),
            "metric_names": tuple(metric_names),
            "fit_time": float(fit_time),
            "predict_time": predict_time,
            "predictions": preds,
            "predict_proba": eval_result["predict_proba"],
            "predict_labels": eval_result["predict_labels"],
            "threshold": eval_result["threshold"],
        })
        last_predictions = preds
        last_metric_names = tuple(metric_names)
        last_predict_proba = eval_result["predict_proba"]
        last_predict_labels = eval_result["predict_labels"]
        last_threshold = eval_result["threshold"]

    # Aggregate
    if seed_num > 1:
        loss_mean, loss_std, m_mean, m_std = _aggregate_metrics(per_seed, last_metric_names)
    else:
        loss_mean = per_seed[0]["loss"]
        loss_std = 0.0
        m_mean = per_seed[0]["metrics"]
        m_std = tuple(0.0 for _ in last_metric_names)

    result = RunResult(
        predictions=last_predictions,
        predict_proba=last_predict_proba,
        predict_labels=last_predict_labels,
        threshold=last_threshold,
        loss=per_seed[-1]["loss"],
        metrics=per_seed[-1]["metrics"],
        metric_names=last_metric_names,
        fit_time=float(np.mean(fit_times)) if fit_times else 0.0,
        predict_time=float(np.mean(predict_times)) if predict_times else 0.0,
        loss_mean=loss_mean,
        loss_std=loss_std,
        metrics_mean=m_mean,
        metrics_std=m_std,
        per_seed=per_seed,
        method_type=model_type,
        config=getattr(args, "config", {}) or {},
        save_path=getattr(args, "save_path", None),
    )
    if return_method:
        result.method = method  # type: ignore[attr-defined]
    return result


# ----------------------------------------------------------------------------
#  Convenience entry points
# ----------------------------------------------------------------------------

def run_from_dataset(
    model_type: str,
    dataset: str,
    dataset_path: str = "./data",
    **kwargs,
) -> RunResult:
    """Load a dataset from disk via :func:`TALENT.model.lib.data.get_dataset`
    and run a method on it. Thin wrapper for the common case.
    """
    from TALENT.model.lib.data import get_dataset
    train_val_data, test_data, info = get_dataset(dataset, dataset_path)
    return run(model_type, train_val_data, test_data, info,
               dataset=dataset, dataset_path=dataset_path, **kwargs)


__all__ = [
    # Result type
    "RunResult",
    # Args / runner
    "build_args",
    "run",
    "run_from_dataset",
    # Registry surface (re-exported for convenience)
    "MethodSpec",
    "METHOD_REGISTRY",
    "Architecture",
    "Hardware",
    "OutputType",
    "get_method_spec",
    "get_method_class",
    "list_methods",
    "deep_method_names",
    "classical_method_names",
    "all_method_names",
]
