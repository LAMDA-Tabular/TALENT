"""Reproducible benchmark splits and matched comparisons."""

from .splits import BenchmarkSplit, make_split
from .report import compare_runs, result_record

__all__ = ["BenchmarkSplit", "make_split", "compare_runs", "result_record"]
