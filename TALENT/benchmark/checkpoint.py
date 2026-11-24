"""Durable, single-writer checkpoints for cross-validation runs."""

import hashlib
import json
import os
import platform
import tempfile
from contextlib import contextmanager
from dataclasses import fields
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


def run_identity(spec, data, protocol, options, root):
    from joblib import hash as hash_input
    from TALENT.model.utils import get_device
    import torch

    options = dict(options)
    args = options.pop("args", None)
    defaults = dict(vars(args)) if args is not None else {}
    defaults.pop("save_path", None)  # Each fit gets its own directory under the run root.
    defaults.pop("hpo_time", None)  # Elapsed time on a previously tuned namespace is an output.
    effective = {**defaults, **options}
    config = options.get("config")
    if config is None:
        config = defaults.get("config", {})
    assets = {}
    general = (config or {}).get("general") or {}
    for key in ("model_path", "model_weight_path", "weights", "inference_config"):
        value = general.get(key)
        if not isinstance(value, (str, os.PathLike)) or not os.fspath(value):
            continue
        asset = Path(value).expanduser().resolve()
        if asset.exists():
            digest = hashlib.sha256()
            files = ((asset,) if asset.is_file() else
                     sorted(p for p in asset.rglob("*") if p.is_file() and root not in p.parents))
            for path in files:
                digest.update(path.relative_to(asset).as_posix().encode() + b"\0")
                contents = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        contents.update(chunk)
                digest.update(contents.digest())
            assets[key] = digest.hexdigest()

    requested = effective.get("device")
    if requested is None and effective.get("gpu") in ("cpu", ""):
        requested = "cpu"
    device = get_device(requested)
    hardware = [platform.system(), platform.machine(), platform.processor(), str(device)]
    if device.type == "cuda":
        hardware.extend([torch.cuda.get_device_name(device), torch.version.cuda])

    package = Path(__file__).resolve().parents[1]
    source = hashlib.sha256()
    for path in sorted(p for p in package.rglob("*")
                       if p.suffix in (".py", ".json") and root not in p.parents):
        source.update(path.relative_to(package).as_posix().encode())
        source.update(hashlib.sha256(path.read_bytes()).digest())
    packages = {}
    for name in ("numpy", "scipy", "pandas", "torch", "scikit-learn", "optuna",
                 spec.backend_distribution, spec.name if spec.name in ("xgboost", "catboost", "lightgbm") else None):
        if name:
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
    return {"format": 1, "method": spec.name, "data": hash_input(data, hash_name="sha1"),
            "protocol": hash_input(protocol, hash_name="sha1"),
            "options": hash_input((defaults, options), hash_name="sha1"), "assets": assets,
            "source": source.hexdigest(), "packages": packages, "hardware": hardware}


@contextmanager
def atomic_file(path):
    tmp = tempfile.NamedTemporaryFile(dir=path.parent, prefix=".talent-tmp-", delete=False)
    try:
        with tmp:
            yield tmp
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp.name, path)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@contextmanager
def run_directory(method, parent, resume_dir, identity):
    if resume_dir is None:
        if parent is not None:
            Path(parent).mkdir(parents=True, exist_ok=True)
        yield Path(tempfile.mkdtemp(prefix=f"talent-cv-{method}-", dir=parent))
        return

    from filelock import FileLock, Timeout

    root = Path(resume_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(root / ".talent.lock"))
    try:
        lock.acquire(timeout=0)
    except Timeout as exc:
        raise RuntimeError(f"Another worker is using {root}") from exc
    try:
        manifest = root / "run.json"
        current = identity(root)
        if manifest.exists():
            recorded = json.loads(manifest.read_text(encoding="utf-8"))
            if recorded != current:
                changed = [key for key in current if recorded.get(key) != current[key]]
                raise ValueError(f"Incompatible resume directory: changed {', '.join(changed)}")
        else:
            if any(p.name != ".talent.lock" and not p.name.startswith(".talent-tmp-") for p in root.iterdir()):
                raise ValueError("Choose an empty resume directory or an existing TALENT run")
            with atomic_file(manifest) as out:
                out.write(json.dumps(current, sort_keys=True).encode())
        yield root
    finally:
        lock.release()


_ARRAYS = ("predictions", "predict_proba", "predict_labels")
_TUPLES = ("metrics", "metric_names", "metrics_mean", "metrics_std")


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return os.fspath(value)
    raise TypeError(f"Cannot checkpoint {type(value).__name__} as JSON")


def _pack(value):
    """Preserve configuration containers, including integer class-weight keys."""
    if isinstance(value, dict):
        return ["dict", [[_pack(k), _pack(v)] for k, v in value.items()]]
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [_pack(v) for v in value]]
    if isinstance(value, Path):
        return ["path", str(value)]
    if isinstance(value, np.ndarray) and value.dtype.kind in "biufU":
        return ["array", [value.dtype.str, value.shape, _pack(value.tolist())]]
    return ["value", value]


def _unpack(record):
    kind, value = record
    if kind == "dict":
        return {_unpack(k): _unpack(v) for k, v in value}
    if kind == "tuple":
        return tuple(_unpack(v) for v in value)
    if kind == "list":
        return [_unpack(v) for v in value]
    if kind == "path":
        return Path(value)
    if kind == "array":
        dtype, shape, elements = value
        return np.asarray(_unpack(elements), dtype=dtype).reshape(shape)
    if kind != "value":
        raise ValueError(f"Unknown checkpoint value type {kind!r}")
    return value


def save_result(path, result, seconds):
    metadata = {f.name: getattr(result, f.name) for f in fields(result)
                if f.name not in (*_ARRAYS, "per_seed")}
    metadata["per_seed"] = []
    metadata["config"] = _pack(metadata["config"])
    metadata["provenance"] = _pack(metadata["provenance"])
    arrays = {}
    for i, seed in enumerate(result.per_seed):
        metadata["per_seed"].append({k: v for k, v in seed.items() if k not in _ARRAYS})
        for name in _ARRAYS:
            if seed[name] is not None:
                array = np.asarray(seed[name])
                if array.dtype.hasobject:
                    raise ValueError("Cannot checkpoint object-valued predictions")
                arrays[f"{i}_{name}"] = array
    payload = json.dumps({"result": metadata, "seconds": seconds, "arrays": list(arrays)}, default=_json_value)
    with atomic_file(path) as out:
        np.savez(out, metadata=payload, **arrays)


def load_result(path, split_id):
    from TALENT.api import RunResult

    with np.load(path, allow_pickle=False) as archive:
        payload = json.loads(archive["metadata"].item())
        data = payload["result"]
        expected = set(payload["arrays"])
        if set(archive.files) != {"metadata", *expected}:
            raise ValueError(f"Incomplete benchmark result in {path}")
        data["config"] = _unpack(data["config"])
        data["provenance"] = _unpack(data["provenance"])
        if data["split_id"] != split_id:
            raise ValueError(f"Unexpected split in {path}")
        for key in _TUPLES:
            data[key] = tuple(data[key])
        for i, seed in enumerate(data["per_seed"]):
            for name in _ARRAYS:
                key = f"{i}_{name}"
                seed[name] = archive[key] if name == "predictions" or key in expected else None
                if seed[name] is not None and len(seed[name]) != seed["query_rows"]:
                    raise ValueError(f"Unexpected prediction length in {path}")
            for key in ("metrics", "metric_names"):
                seed[key] = tuple(seed[key])
        data.update({name: data["per_seed"][-1][name] for name in _ARRAYS})
    return RunResult(**data), payload["seconds"]
