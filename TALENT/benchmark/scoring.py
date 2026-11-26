"""User-defined metrics for tabular predictions and decision thresholds."""

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np


@dataclass(frozen=True)
class Scorer:
    """A named scalar metric of (y_true, response), with an explicit direction.

    Classification labels are integer class indices in sorted training-class
    order. "proba" supplies all probability columns in that same order.
    "prediction" supplies regression predictions in original target units.
    """

    name: str
    func: Callable
    response: Literal["labels", "proba", "prediction"]
    greater_is_better: bool

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip() or not callable(self.func):
            raise ValueError("Scorer requires a nonempty name and a callable")
        if self.response not in ("labels", "proba", "prediction"):
            raise ValueError("response must be labels, proba or prediction")
        if not isinstance(self.greater_is_better, bool):
            raise ValueError("greater_is_better must be a boolean")

    def __call__(self, y_true, response):
        value = np.asarray(self.func(y_true, response))
        if value.ndim != 0 or value.dtype.kind not in "fiu" or not np.isfinite(value):
            raise ValueError(f"Scorer {self.name!r} must return a finite scalar")
        return float(value)
