"""Cross-validation using the same adapters, evaluator and row manifests as a single run."""

import tempfile
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from numbers import Integral
from pathlib import Path
from time import perf_counter

import numpy as np

from .splits import BenchmarkSplit, _ids, _unit_codes


def _fold_count(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 2:
        raise ValueError(f"{name} must be an integer of at least two")


def make_folds(row_ids, *, cv=5, val_fraction=0.2, seed=0, stratify=None,
               groups=None, timestamps=None, gap=0):
    """Build outer test folds, each with its own training and validation partition.

    Grouped folds split whole groups. Temporal folds expand forward through
    distinct timestamps. The gap counts timestamp blocks at both boundaries.
    """
    from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, train_test_split

    ids = np.asarray(_ids(row_ids, "dataset"), dtype=object)
    _fold_count(cv, "cv")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    if sum(x is not None for x in (groups, timestamps, stratify)) > 1:
        raise ValueError("choose one of stratify, groups or timestamps")
    if isinstance(gap, bool) or not isinstance(gap, Integral) or gap < 0 or (gap and timestamps is None):
        raise ValueError("gap must be a nonnegative integer and requires timestamps when positive")

    temporal = timestamps is not None
    codes = np.arange(len(ids))
    if temporal or groups is not None:
        codes = _unit_codes(timestamps if temporal else groups, len(ids), temporal)
    units = np.arange(codes.max() + 1)
    labels = None
    if temporal:
        strategy = "temporal"
        partitions = TimeSeriesSplit(n_splits=cv, gap=gap).split(units)
    elif stratify is not None:
        strategy = "random"
        labels = np.asarray(stratify)
        if labels.shape != ids.shape:
            raise ValueError("stratify requires one label per row")
        if np.unique(labels, return_counts=True)[1].min() < cv:
            raise ValueError("stratified CV requires at least cv rows in every class")
        partitions = StratifiedKFold(cv, shuffle=True, random_state=seed).split(units, labels)
    else:
        strategy = "group" if groups is not None else "random"
        partitions = KFold(cv, shuffle=True, random_state=seed).split(units)

    # Preserve both the unit order and the original order within tied units.
    ordered = np.argsort(codes, kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(codes))]

    def members(selection):
        if temporal or groups is not None:
            selection = np.concatenate([ordered[offsets[u]:offsets[u + 1]] for u in selection])
        return tuple(ids[selection])

    folds = []
    for development, test in partitions:
        if temporal:
            boundary = len(development) - int(np.ceil(len(development) * val_fraction))
            if boundary <= gap:
                raise ValueError("temporal fold has too little training history for validation and gap")
            train, val = development[:boundary - gap], development[boundary:]
        else:
            train, val = train_test_split(development, test_size=val_fraction, random_state=seed,
                                          stratify=None if labels is None else labels[development])
        train_ids = members(train)
        folds.append(BenchmarkSplit(train_ids, members(val), members(test), train_ids, strategy))
    return tuple(folds)


@dataclass
class CrossValidationResult:
    """One ordinary RunResult per outer fold, in the same order as its split."""

    splits: tuple
    results: tuple
    threshold_splits: tuple
    fold_seconds: tuple
    save_path: str

    @property
    def metrics_mean(self):
        """Equal-fold means of the seed-averaged metrics. Undefined values propagate."""
        scores = np.mean([result.metrics_mean for result in self.results], axis=0)
        return dict(zip(self.results[0].metric_names, scores.tolist()))

    def out_of_fold(self, field="predictions"):
        """Return scored row IDs and last-seed outputs in matching fold order."""
        if field not in ("predictions", "predict_proba", "predict_labels"):
            raise ValueError("choose predictions, predict_proba or predict_labels")
        values = [getattr(result, field) for result in self.results]
        if any(value is None for value in values):
            raise ValueError(f"{field} unavailable for this method and task")
        return tuple(key for split in self.splits for key in split.test), np.concatenate(values)


def cross_validate(method, row_ids, N, C, y, info, *, cv=5, split_seed=0, val_fraction=0.2,
                   groups=None, timestamps=None, gap=0, threshold_cv=None, save_path=None, **options):
    """Evaluate one method with TALENT.run on disjoint outer test folds.

    Classification uses stratification for ordinary shuffled folds. Supply
    groups or timestamps to preserve their boundaries at every level. Optional
    threshold_cv selects a binary threshold per seed from inner out-of-fold
    probabilities. HPO and early stopping use each fit's validation partition.
    """
    from TALENT import get_method_spec
    from TALENT.model.method_registry import OutputType
    from TALENT.model.lib.threshold import tune_threshold
    from TALENT.model.lib.tuning_metric import validate_tune_metric

    spec = get_method_spec(method)
    ids = _ids(row_ids, "dataset")
    labels = np.asarray(y)
    if labels.shape == (len(ids), 1):
        labels = labels[:, 0]
    if labels.shape != (len(ids),):
        raise ValueError("provide one target per row ID")
    classification = info["task_type"] != "regression"
    if threshold_cv is not None:
        _fold_count(threshold_cv, "threshold_cv")
        if (info["task_type"] != "binclass" or spec.output_type == OutputType.CLASS_LABELS
                or not options.get("tune_threshold", True) or options.get("decision_threshold") is not None):
            raise ValueError("threshold_cv requires binary probability predictions and threshold tuning")

    positions = {key: i for i, key in enumerate(ids)}
    groups = None if groups is None else np.asarray(groups)
    timestamps = None if timestamps is None else np.asarray(timestamps)

    def plan(indices, count):
        return make_folds(
            [ids[i] for i in indices], cv=count, seed=split_seed, val_fraction=val_fraction, gap=gap,
            stratify=labels[indices] if classification and groups is None and timestamps is None else None,
            groups=None if groups is None else groups[indices],
            timestamps=None if timestamps is None else timestamps[indices],
        )

    if N is None and C is None:
        raise ValueError("provide numerical or categorical features")
    for values in (N, C, groups, timestamps):
        if values is not None and len(values) != len(ids):
            raise ValueError("features, groups and timestamps must align with row IDs")
    folds = plan(np.arange(len(ids)), cv)
    nested = []
    for outer in folds:
        inner = ()
        if threshold_cv is not None:
            dev = [positions[key] for key in outer.train + outer.val]
            inner = plan(dev, threshold_cv)
        nested.append(inner)

    args = options.get("args")
    tune = options.get("tune")
    tune = getattr(args, "tune", False) if tune is None else tune
    metric = options.get("tune_metric") or getattr(args, "tune_metric", None)
    objective = validate_tune_metric(metric) if tune else None
    regression_metric = objective in ("MAE", "RMSE", "R2")
    if objective and regression_metric != (info["task_type"] == "regression"):
        raise ValueError(f"HPO metric {objective} is unavailable for {info['task_type']}")

    classes = np.unique(labels) if classification else None
    if info["task_type"] == "binclass" and len(classes) != 2:
        raise ValueError("binclass requires exactly two classes")
    for split in chain(folds, chain.from_iterable(nested)):
        if classification:
            known = np.unique(labels[[positions[key] for key in split.train]])
            if not np.array_equal(known, classes):
                raise ValueError("a training fold lacks a class; change the split sizes or history")
        if objective == "AUC":
            val_classes = np.unique(labels[[positions[key] for key in split.val]])
            if not np.array_equal(val_classes, classes):
                raise ValueError("AUC tuning requires all classes in every validation fold")
        if objective == "R2" and len(split.val) < 2:
            raise ValueError("R2 tuning requires at least two rows in every validation fold")

    parent = save_path if save_path is not None else getattr(args, "save_path", None)
    if parent is not None:
        Path(parent).mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"talent-cv-{method}-", dir=parent))

    def fit(split, directory, overrides):
        kwargs = {**options, **overrides}
        if kwargs.get("args") is not None:
            kwargs["args"] = deepcopy(kwargs["args"])
        return split.run(method, ids, N, C, labels, info, save_path=str(directory),
                         allow_extra_rows=True, **kwargs)

    results, fold_seconds = [], []
    for i, (outer, inner) in enumerate(zip(folds, nested)):
        started = perf_counter()
        directory = root / f"fold-{i}"
        overrides = {}
        if inner:
            targets = []
            per_seed = {}
            for j, split in enumerate(inner):
                run = fit(split, directory / f"inner-{j}", {"tune_threshold": False, "return_method": False})
                targets.extend(labels[[positions[key] for key in split.test]] == classes[1])
                for seed_run in run.per_seed:
                    proba = seed_run["predict_proba"]
                    if proba is None or proba.shape != (len(split.test), 2) or not np.isfinite(proba).all():
                        raise ValueError("inner threshold selection requires finite binary probabilities")
                    per_seed.setdefault(seed_run["seed"], []).append(proba[:, 1])
            if len(np.unique(targets)) < 2:
                raise ValueError("inner threshold predictions must represent both classes")
            overrides["decision_threshold"] = {
                seed: tune_threshold(targets, np.concatenate(scores), metric=options.get("threshold_metric", "f1"))[0]
                for seed, scores in per_seed.items()
            }
        results.append(fit(outer, directory / "outer", overrides))
        fold_seconds.append(perf_counter() - started)
    return CrossValidationResult(folds, tuple(results), tuple(nested), tuple(fold_seconds), str(root))


def cross_validate_from_dataset(method, dataset, dataset_path="./data", **options):
    """Cross-validate the original train and validation rows of a TALENT-format dataset."""
    from TALENT.model.lib.data import get_dataset

    train_val, _, info = get_dataset(dataset, dataset_path)
    ids = [f"{part}/{i}" for part in ("train", "val") for i in range(len(train_val[2][part]))]
    N, C, y = (None if values is None else np.concatenate((values["train"], values["val"]))
               for values in train_val)
    return cross_validate(method, ids, N, C, y, info, dataset=dataset, dataset_path=dataset_path,
                          **options)
