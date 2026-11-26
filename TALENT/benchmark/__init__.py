"""Reproducible benchmark splits and matched comparisons."""

from .splits import BenchmarkSplit, make_split, shuffled_holdout_indices, subsample_fold_training_indices
from .report import compare_runs, result_record
from .cv import CrossValidationResult, cross_validate, cross_validate_from_dataset, iter_shuffled_fold_indices, make_folds

__all__ = ["BenchmarkSplit", "make_split", "make_folds", "cross_validate",
           "shuffled_holdout_indices", "iter_shuffled_fold_indices", "subsample_fold_training_indices",
           "cross_validate_from_dataset", "CrossValidationResult", "compare_runs", "result_record"]
