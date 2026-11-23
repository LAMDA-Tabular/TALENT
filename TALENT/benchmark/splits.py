"""Persist ordered row identifiers for replayable TALENT experiments."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import date
from numbers import Integral, Real
from pathlib import Path

import numpy as np


def _ids(values, name):
    if isinstance(values, (str, bytes, set, frozenset, dict)):
        raise ValueError(f"{name} must be an ordered sequence of row IDs")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an ordered sequence of row IDs") from exc
    if not items or any(not isinstance(x, str) or not x for x in items):
        raise ValueError(f"{name} requires nonempty string row IDs")
    if len(set(items)) != len(items):
        raise ValueError(f"{name} contains duplicate row IDs")
    return items


@dataclass(frozen=True)
class BenchmarkSplit:
    """An ordered partition and the subset of training rows supplied to the method."""

    train: tuple
    val: tuple
    test: tuple
    context: tuple
    strategy: str = "explicit"

    def __post_init__(self):
        for name in ("train", "val", "test", "context"):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        _ids(self.train + self.val + self.test, "partition")
        if not set(self.context).issubset(self.train):
            raise ValueError("context must contain only training IDs")
        if self.strategy not in ("explicit", "random", "group", "temporal"):
            raise ValueError("unknown split strategy")

    def _payload(self):
        return {"version": 1, **asdict(self)}

    @property
    def split_id(self):
        """SHA-256 of the ordered partitions, context and split strategy."""
        encoded = json.dumps(self._payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def with_context(self, size):
        """Choose a nested training subset while preserving validation and test IDs."""
        if isinstance(size, bool) or not isinstance(size, Integral) or not 1 <= size <= len(self.train):
            raise ValueError("context size must be an integer between 1 and the training size")
        context = self.train[-size:] if self.strategy == "temporal" else self.train[:size]
        return replace(self, context=context)

    def save(self, path):
        """Atomically write a versioned JSON manifest. IDs are stored in plain text."""
        path = Path(path)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             delete=False) as out:
                tmp = out.name
                json.dump({**self._payload(), "split_id": self.split_id}, out)
            os.replace(tmp, path)
        finally:
            if tmp is not None:
                Path(tmp).unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.pop("version", None) != 1:
            raise ValueError("unsupported split manifest version")
        expected = manifest.pop("split_id", None)
        if set(manifest) != {"train", "val", "test", "context", "strategy"}:
            raise ValueError("split manifest has missing or unrecognized fields")
        split = cls(**manifest)
        if expected != split.split_id:
            raise ValueError("split manifest fingerprint mismatch")
        return split

    def materialize(self, row_ids, N, C, y):
        """Return TALENT train/validation and test tuples from row-aligned arrays."""
        ids = _ids(row_ids, "dataset")
        if set(ids) != set(self.train + self.val + self.test):
            raise ValueError("dataset row IDs differ from the saved partition")
        if y is None or (N is None and C is None):
            raise ValueError("provide targets and at least one numerical or categorical feature array")
        positions = {key: i for i, key in enumerate(ids)}
        selections = {part: [positions[key] for key in keys] for part, keys in
                      (("train", self.context), ("val", self.val), ("test", self.test))}
        training, testing = [], []
        for values in (N, C, y):
            if values is None:
                training.append(None)
                testing.append(None)
                continue
            array = np.asarray(values)
            if array.ndim == 0 or len(array) != len(ids):
                raise ValueError("all arrays must have one entry per dataset row ID")
            training.append({part: array[selections[part]] for part in ("train", "val")})
            testing.append({"test": array[selections["test"]]})
        return tuple(training), tuple(testing)

    def run(self, method, row_ids, N, C, y, info, *, capture_support_ids=False, **options):
        """Replay this split through TALENT.run and attach its fingerprint."""
        from TALENT import run
        from .measurement import tracking_support

        if capture_support_ids and not options.get("measure"):
            raise ValueError("capture_support_ids requires measure=True")
        train_val, test = self.materialize(row_ids, N, C, y)
        with tracking_support(self.context if capture_support_ids else None):
            result = run(method, train_val, test, info, **options)
        result.split_id = self.split_id
        return result


def make_split(row_ids, *, val_fraction=0.2, test_fraction=0.2, seed=0,
               groups=None, timestamps=None):
    """Split rows, whole groups, or whole timestamp blocks into train/val/test.

    Fractions apply to the number of split units. Temporal partitions are
    strictly ordered. Random and group splits randomize training-row order
    for nested context prefixes. Supply stable, non-sensitive row IDs.
    """
    import pandas as pd

    ids = _ids(row_ids, "dataset")
    if groups is not None and timestamps is not None:
        raise ValueError("choose either grouped or temporal splitting")
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("validation and test fractions must be positive and leave training data")
    rng = np.random.default_rng(seed)
    strategy = "random"
    if groups is None and timestamps is None:
        codes = np.arange(len(ids))
        units = rng.permutation(len(ids))
    else:
        temporal = timestamps is not None
        values = pd.Index(timestamps if temporal else groups)
        if len(values) != len(ids) or values.hasnans:
            raise ValueError("supply one nonmissing group or timestamp per row")
        if temporal:
            numeric = all(isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
                          and np.isfinite(value) for value in values)
            if not numeric:
                if not all(isinstance(value, (date, np.datetime64)) for value in values):
                    raise ValueError("timestamps must all be finite numbers or all be datetimes")
                values = pd.DatetimeIndex(values)
        codes, unique = pd.factorize(values, sort=temporal)
        units = np.arange(len(unique)) if temporal else rng.permutation(len(unique))
        strategy = "temporal" if temporal else "group"

    n_val = int(np.ceil(len(units) * val_fraction))
    n_test = int(np.ceil(len(units) * test_fraction))
    n_train = len(units) - n_val - n_test
    if n_train < 1:
        raise ValueError("too few split units for three nonempty partitions")

    # Order rows within partitions as well as assigning the partition itself.
    if strategy == "temporal":
        order = np.argsort(codes, kind="stable")
    else:
        order = rng.permutation(len(ids))

    def members(block):
        rows = order[np.isin(codes[order], block)]
        return tuple(ids[i] for i in rows)

    train = members(units[:n_train])
    val = members(units[n_train:n_train + n_val])
    test = members(units[n_train + n_val:])
    return BenchmarkSplit(train, val, test, train, strategy)
