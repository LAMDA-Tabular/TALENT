"""Persist ordered row identifiers for replayable TALENT experiments."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import date
from functools import cached_property
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

    @cached_property
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

    def with_holdout_limits(self, *, val_limit=None, test_limit=None,
                            seed=0, stratify=None, timestamps=None):
        """Cap existing holdouts without changing training rows or moving observations.

        Supply a mapping from row ID to class label for stratified sampling, or
        to time for a suffix of the newest whole timestamp blocks. Without
        either mapping, sample uniformly without replacement. Replaying the
        reduced split with the full table requires `allow_extra_rows=True`.
        """
        if stratify is not None and timestamps is not None:
            raise ValueError("choose stratify or timestamps for holdout limits")
        val_seed, test_seed = np.random.SeedSequence(seed).spawn(2)
        val = _limit_rows(self.val, val_limit, np.random.default_rng(val_seed), stratify, timestamps, "val_limit")
        test = _limit_rows(self.test, test_limit, np.random.default_rng(test_seed), stratify, timestamps, "test_limit")
        return replace(self, val=val, test=test) if (val, test) != (self.val, self.test) else self

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

    def materialize(self, row_ids, N, C, y, *, allow_extra_rows=False):
        """Return TALENT train/validation and test tuples from row-aligned arrays."""
        ids = _ids(row_ids, "dataset")
        partition = set(self.train + self.val + self.test)
        positions = {key: i for i, key in enumerate(ids)}
        if not partition.issubset(positions) or (not allow_extra_rows and len(ids) != len(partition)):
            raise ValueError("dataset row IDs differ from the saved partition")
        if y is None or (N is None and C is None):
            raise ValueError("provide targets and at least one numerical or categorical feature array")
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

    def run(self, method, row_ids, N, C, y, info, *, capture_support_ids=False,
            allow_extra_rows=False, **options):
        """Replay this split through TALENT.run and attach its fingerprint."""
        from TALENT import run
        from .measurement import tracking_support

        if capture_support_ids and not options.get("measure"):
            raise ValueError("capture_support_ids requires measure=True")
        train_val, test = self.materialize(row_ids, N, C, y, allow_extra_rows=allow_extra_rows)
        with tracking_support(self.context if capture_support_ids else None):
            result = run(method, train_val, test, info, **options)
        result.split_id = self.split_id
        return result


def _unit_index(values):
    from pandas import Index

    # A tuple can identify one group, just as a string or integer can.
    return Index(values, tupleize_cols=False)


def _unit_codes(values, n_rows, temporal):
    import pandas as pd

    values = _unit_index(values)
    if len(values) != n_rows or values.hasnans:
        raise ValueError("supply one nonmissing group or timestamp per row")
    if temporal:
        numeric = all(isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
                      and np.isfinite(value) for value in values)
        if not numeric:
            if not all(isinstance(value, (date, np.datetime64)) for value in values):
                raise ValueError("timestamps must all be finite numbers or all be datetimes")
            values = pd.DatetimeIndex(values)
    return pd.factorize(values, sort=temporal)[0]


def _row_metadata(ids, values):
    if values is None:
        return None
    values = _unit_index(values)
    if len(values) != len(ids):
        raise ValueError("holdout metadata must align with row IDs")
    return dict(zip(ids, values))


def _split_development(indices, val_fraction, seed, labels):
    from sklearn.model_selection import train_test_split

    return train_test_split(indices, test_size=val_fraction, random_state=seed,
                            stratify=None if labels is None else labels[indices])


def shuffled_holdout_indices(n_rows, *, test_fraction=0.2, val_fraction=0.2,
                             seed=0, validation_seed=None, stratify=None):
    """Return train, validation and test indices with validation relative to development."""
    from sklearn.model_selection import train_test_split

    labels = None if stratify is None else np.asarray(stratify)
    indices = np.arange(n_rows)
    development, test = train_test_split(
        indices, test_size=test_fraction, random_state=seed, stratify=labels,
    )
    train, val = _split_development(
        development, val_fraction, seed if validation_seed is None else validation_seed, labels,
    )
    return train, val, test


def _limit_rows(rows, limit, rng, stratify, timestamps, name):
    if limit is None:
        return rows
    if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 1:
        raise ValueError(f"{name} must be a positive integer or None")
    if len(rows) <= limit:
        return rows
    if stratify is None and timestamps is None:
        selected = rng.choice(len(rows), size=limit, replace=False)
    else:
        metadata = timestamps if timestamps is not None else stratify
        try:
            values = [metadata[row] for row in rows]
        except KeyError as exc:
            raise ValueError(f"{name} requires metadata for every row in its holdout") from exc
        codes = _unit_codes(values, len(rows), temporal=timestamps is not None)
        counts = np.bincount(codes)
        if timestamps is not None:
            blocks = np.searchsorted(np.cumsum(counts[::-1]), limit, side="right")
            if not blocks:
                raise ValueError(f"{name} cannot fit the newest complete timestamp block")
            selected = np.flatnonzero(codes >= len(counts) - blocks)
        else:
            if limit < len(counts):
                raise ValueError(f"{name} cannot represent all {len(counts)} strata")
            # Reserve one row per stratum, then apportion the remaining slots.
            capacity = counts - 1
            remainder = limit - len(counts)
            proportional = capacity * (remainder / capacity.sum())
            extra = proportional.astype(int)
            quota = 1 + extra
            largest = np.argsort(extra - proportional, kind="stable")
            quota[largest[:remainder - extra.sum()]] += 1
            order = rng.permutation(len(rows))
            ordered = order[np.argsort(codes[order], kind="stable")]
            offsets = np.r_[0, np.cumsum(counts)]
            selected = np.concatenate([ordered[start:start + size] for start, size in zip(offsets, quota)])
    return tuple(rows[i] for i in np.sort(selected))


def make_split(row_ids, *, val_fraction=0.2, test_fraction=0.2, seed=0,
               groups=None, timestamps=None, stratify=None, val_limit=None, test_limit=None):
    """Split rows, whole groups, or whole timestamp blocks into train/val/test.

    Fractions apply to the number of split units. Temporal partitions are
    strictly ordered. Random and group splits randomize training-row order
    for nested context prefixes. Pass one class label per row as stratify
    for a stratified random split. Supply stable, non-sensitive row IDs.
    Optional row limits reduce the resulting holdouts using the same metadata.
    """
    ids = _ids(row_ids, "dataset")
    if sum(value is not None for value in (groups, timestamps, stratify)) > 1:
        raise ValueError("choose one of stratify, groups or timestamps")
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("validation and test fractions must be positive and leave training data")

    def limit_holdouts(split):
        if val_limit is None and test_limit is None:
            return split
        return split.with_holdout_limits(
            val_limit=val_limit, test_limit=test_limit, seed=seed,
            stratify=_row_metadata(ids, stratify), timestamps=_row_metadata(ids, timestamps),
        )
    if stratify is not None:
        from sklearn.model_selection import train_test_split

        labels = np.asarray(stratify)
        if labels.shape != (len(ids),):
            raise ValueError("stratify requires one class label per row")
        rows = np.arange(len(ids))
        n_test = int(np.ceil(len(ids) * test_fraction))
        n_val = int(np.ceil(len(ids) * val_fraction))
        if n_test + n_val >= len(ids):
            raise ValueError("too few rows for three nonempty partitions")
        development, test = train_test_split(
            rows, test_size=n_test, random_state=seed, stratify=labels
        )
        train, val = train_test_split(
            development, test_size=n_val, random_state=seed, stratify=labels[development]
        )
        classes = np.unique(labels)
        if any(not np.array_equal(np.unique(labels[part]), classes) for part in (train, val, test)):
            raise ValueError("every stratified partition needs all classes; adjust the fractions or supply more rows")

        def row_ids_for(positions):
            return tuple(ids[i] for i in positions)

        train_ids = row_ids_for(train)
        return limit_holdouts(BenchmarkSplit(train_ids, row_ids_for(val), row_ids_for(test), train_ids, "random"))

    rng = np.random.default_rng(seed)
    strategy = "random"
    if groups is None and timestamps is None:
        codes = np.arange(len(ids))
        units = rng.permutation(len(ids))
    else:
        temporal = timestamps is not None
        codes = _unit_codes(timestamps if temporal else groups, len(ids), temporal)
        count = codes.max() + 1
        units = np.arange(count) if temporal else rng.permutation(count)
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
    return limit_holdouts(BenchmarkSplit(train, val, test, train, strategy))
