"""Dataset-balanced summaries of benchmark coverage, paired quality and cost."""

from collections import Counter, defaultdict
from math import isfinite, sqrt
from statistics import mean, stdev


def result_record(result, dataset, metric, *, split_id=None):
    """Extract one scalar record from a completed run (averaged across its seeds)."""
    if split_id is not None and result.split_id is not None and split_id != result.split_id:
        raise ValueError("split_id differs from the split used to run the method")
    split_id = result.split_id or split_id
    if not split_id:
        raise ValueError("supply a split_id or run through BenchmarkSplit.run")
    scores = dict(zip(result.metric_names, result.metrics_mean))
    if metric not in scores:
        raise ValueError(f"metric {metric!r} is unavailable; choose from {tuple(scores)}")
    seeds = result.measurements.get("seeds", [])
    if not seeds and result.per_seed and all(s.get("query_rows") for s in result.per_seed):
        seeds = [{"fit_seconds": s["fit_time"],
                  "predict_seconds_per_1000": s["predict_time"] * 1000 / s["query_rows"]}
                 for s in result.per_seed]
    cost = {}
    for key in ("fit_seconds", "predict_seconds_per_1000"):
        values = [s.get(key) for s in seeds]
        cost[key] = mean(values) if values and all(v is not None for v in values) else None
    return {"dataset": dataset, "split_id": split_id, "method": result.method_type,
            "status": "ok", "metric": metric, "score": float(scores[metric]), **cost}


def _names(values, label):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"provide a collection of {label} names")
    names = list(values)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{label} names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate {label} names")
    return names


def _dataset_means(values):
    grouped = defaultdict(list)
    for (dataset, _), value in values.items():
        grouped[dataset].append(value)
    return {dataset: mean(scores) for dataset, scores in grouped.items()}


def _finite_number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if isfinite(number) else None


def compare_runs(records, *, expected, methods, metric, reference,
                 higher_is_better=True, confidence=0.95):
    """Compare one metric on an explicit dataset-to-split-ID schedule.

    Rankings use the intersection across all methods. Each reference contrast
    uses its own pairwise intersection. Both average splits within dataset
    before weighting datasets equally. Positive differences favor the method.
    """
    from scipy.stats import rankdata, t

    methods = _names(methods, "method")
    if not isinstance(metric, str) or not metric:
        raise ValueError("supply a metric name")
    if reference not in methods or not 0 < confidence < 1:
        raise ValueError("choose a listed reference and confidence between zero and one")
    cells = [(dataset, split) for dataset in _names(expected, "dataset")
             for split in _names(expected[dataset], "split")]
    schedule = set(cells)
    index = {}
    for record in records:
        cell = (record["dataset"], record["split_id"])
        method = record["method"]
        if cell not in schedule or method not in methods:
            raise ValueError(f"unscheduled benchmark record: {(cell, method)!r}")
        key = (cell, method)
        if key in index:
            raise ValueError(f"duplicate benchmark record: {key!r}")
        status = record["status"]
        if status not in ("ok", "failed", "skipped"):
            raise ValueError(f"unknown benchmark status: {status!r}")
        if status == "ok":
            if record.get("metric") != metric:
                raise ValueError(f"expected metric {metric!r} for {key!r}")
            if _finite_number(record.get("score")) is None:
                status = "invalid"
        index[key] = (status, record)

    coverage, counts = [], {name: Counter() for name in methods}
    success = {method: {} for method in methods}
    for cell in cells:
        for method in methods:
            status, record = index.get((cell, method), ("missing", {}))
            counts[method][status] += 1
            coverage.append({"dataset": cell[0], "split_id": cell[1], "method": method,
                             "status": status, "reason": record.get("reason")})
            if status == "ok":
                success[method][cell] = float(record["score"])

    common = [cell for cell in cells if all(cell in success[m] for m in methods)]
    direction = 1 if higher_is_better else -1
    ranking = []
    for method in methods:
        quality = _dataset_means({cell: success[method][cell] for cell in common})
        row = {"method": method, "score": mean(quality.values()) if quality else None,
               "datasets": len(quality), "splits": len(common), "rank": None}
        for field in ("fit_seconds", "predict_seconds_per_1000"):
            measured = {}
            for cell in common:
                value = _finite_number(index[cell, method][1].get(field))
                if value is not None and value >= 0:
                    measured[cell] = value
            row[field + "_splits"] = len(measured)
            row[field] = (mean(_dataset_means(measured).values())
                          if common and len(measured) == len(common) else None)
        ranking.append(row)
    if common:
        for row, rank in zip(ranking, rankdata([-direction * r["score"] for r in ranking])):
            row["rank"] = float(rank)
        ranking.sort(key=lambda row: row["rank"])

    contrasts = []
    for method in methods:
        if method == reference:
            continue
        paired = {cell: direction * (score - success[reference][cell])
                  for cell, score in success[method].items() if cell in success[reference]}
        differences = _dataset_means(paired)
        n = len(differences)
        delta = mean(differences.values()) if n else None
        interval = None
        if n >= 2:
            half_width = float(t.ppf((1 + confidence) / 2, n - 1)) * stdev(differences.values()) / sqrt(n)
            interval = [delta - half_width, delta + half_width]
        contrasts.append({"method": method, "reference": reference, "splits": len(paired),
                          "dataset_differences": differences, "datasets": n,
                          "mean_difference": delta, "confidence_interval": interval})
    return {"metric": metric, "higher_is_better": higher_is_better, "confidence": confidence,
            "coverage": coverage, "counts": {k: dict(v) for k, v in counts.items()},
            "ranking": ranking, "contrasts": contrasts}
