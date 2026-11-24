"""Reproducible benchmark splits and matched comparisons."""

from .splits import BenchmarkSplit, make_split
from .report import compare_runs, result_record
from .cv import CrossValidationResult, cross_validate, cross_validate_from_dataset, make_folds

__all__ = ["BenchmarkSplit", "make_split", "make_folds", "cross_validate",
           "cross_validate_from_dataset", "CrossValidationResult", "compare_runs", "result_record"]
