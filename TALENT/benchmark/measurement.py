"""Opt-in wall-clock and process-memory observations for TALENT.run."""

import hashlib
import json
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from importlib import metadata
from pathlib import Path

_ACTIVE = ContextVar("talent_run_measurement", default=None)
_ROW_IDS = ContextVar("talent_benchmark_row_ids", default=None)


@contextmanager
def tracking_support(ids):
    token = _ROW_IDS.set(ids)
    try:
        yield
    finally:
        _ROW_IDS.reset(token)


def support_ids():
    return _ROW_IDS.get() if _ACTIVE.get() is not None else None


def timed_call(field, function, *args, **kwargs):
    """Time a phase when the current run requested measurements."""
    meter = _ACTIVE.get()
    if meter is None:
        return function(*args, **kwargs)
    return elapsed_call(field, meter.device, function, *args, **kwargs)[0]


def elapsed_call(field, device, function, /, *args, **kwargs):
    """Return the result and wall time, including outstanding work on the selected GPU."""
    import torch

    gpu = torch.device(device) if device is not None else None
    sync_cuda = gpu is not None and gpu.type == "cuda" and torch.cuda.is_available()
    if sync_cuda:
        torch.cuda.synchronize(gpu)
    started = time.perf_counter()
    result = function(*args, **kwargs)
    if sync_cuda:
        torch.cuda.synchronize(gpu)
    elapsed = time.perf_counter() - started
    meter = _ACTIVE.get()
    if meter is not None:
        meter.data[field] = (meter.data.get(field) or 0.0) + elapsed
        meter.sample()
    return result, elapsed


def record_support(rows, selected_ids=None):
    meter = _ACTIVE.get()
    if meter is not None:
        meter.data["support_rows"] = int(rows)
        if selected_ids is not None:
            meter.data["support_ids"] = selected_ids


def record_query_context(rows, indices=None):
    meter = _ACTIVE.get()
    if meter is not None and meter.data.get("predict_seconds") is None:
        meter.data["context_rows_per_query"] = int(rows)
        ids = meter.data.get("support_ids")
        if ids is not None:
            meter.data["query_context_ids"] = ids if indices is None else [ids[i] for i in indices.tolist()]


def _backend_commit(distribution):
    if not distribution:
        return None
    try:
        direct = metadata.distribution(distribution).read_text("direct_url.json")
        return json.loads(direct or "{}").get("vcs_info", {}).get("commit_id")
    except (metadata.PackageNotFoundError, ValueError, OSError):
        return None


def _model_device(model):
    """Inspect fitted state, without treating the requested device as an observation."""
    import torch

    state = getattr(model, "__dict__", {})
    devices = set()
    for candidate in (model, *state.values()):
        if isinstance(candidate, torch.nn.Module):
            devices.update(str(t.device) for t in candidate.parameters())
            devices.update(str(t.device) for t in candidate.buffers())
    if devices:
        return ",".join(sorted(devices))
    if type(model).__module__.startswith("sklearn."):
        import numpy as np
        if any(isinstance(state.get(name), np.ndarray)
               for name in ("coef_", "_fit_X", "class_prior_", "cluster_centers_")):
            return "cpu"
    return None


def _file_digest(path):
    if path is None:
        return None
    digest = hashlib.sha256()
    try:
        if not Path(path).is_file():
            return None
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


class _Meter:
    def __init__(self, device):
        import torch

        self.cuda = torch.cuda if device.type == "cuda" and torch.cuda.is_available() else None
        self.torch_version = str(torch.__version__)
        self.device = device
        self.rss = None
        try:
            import psutil
            try:
                self.rss = psutil.Process()
                self.rss_errors = (psutil.Error, OSError)
            except (psutil.Error, OSError):
                pass
        except ImportError:
            pass
        self.data = dict.fromkeys(("load_seconds", "support_rows", "context_rows_per_query",
                                   "sampled_peak_rss_bytes", "sampled_peak_cuda_bytes"))
        self._lock = threading.Lock()

    def sample(self):
        with self._lock:
            if self.rss is not None:
                try:
                    current = self.rss.memory_info().rss
                    self.data["sampled_peak_rss_bytes"] = max(self.data["sampled_peak_rss_bytes"] or 0, current)
                except self.rss_errors:
                    self.rss = None
            if self.cuda is not None:
                current = self.cuda.memory_allocated(self.device)
                self.data["sampled_peak_cuda_bytes"] = max(self.data["sampled_peak_cuda_bytes"] or 0, current)

    def finish(self, method, spec, query_rows, hash_checkpoint):
        from os import fspath

        checkpoint = getattr(method, "resolved_checkpoint", None)
        checkpoint = fspath(checkpoint) if checkpoint is not None else None
        snapshot = None
        if checkpoint is not None:
            parts = Path(checkpoint).parts
            snapshot = next((b for a, b in zip(parts, parts[1:])
                             if a == "snapshots" and re.fullmatch("[0-9a-fA-F]{40}", b)), None)
        options = method.args.config.get("general") or {}
        self.data.update(
            method=spec.name, seed=int(method.args.seed), query_rows=query_rows,
            configured_device=str(self.device), model_state_device=_model_device(getattr(method, "model", None)),
            torch_version=self.torch_version,
            cuda_device_name=self.cuda.get_device_name(self.device) if self.cuda is not None else None,
            backend_distribution=spec.backend_distribution, backend_version=spec.backend_version(),
            backend_commit=_backend_commit(spec.backend_distribution), checkpoint=checkpoint,
            checkpoint_revision=snapshot, requested_checkpoint_revision=options.get("revision"),
            checkpoint_sha256=_file_digest(checkpoint) if hash_checkpoint else None,
            predict_seconds_per_1000=self.data["predict_seconds"] * 1000 / query_rows if query_rows else None,
        )


@contextmanager
def measuring(device, enabled):
    if not enabled:
        token = _ACTIVE.set(None)
        try:
            yield None
        finally:
            _ACTIVE.reset(token)
        return
    meter = _Meter(device)
    stop = threading.Event()

    def sample_periodically():
        while not stop.wait(0.02):
            meter.sample()

    meter.sample()
    worker = threading.Thread(target=sample_periodically, daemon=True)
    worker.start()
    token = _ACTIVE.set(meter)
    try:
        yield meter
    finally:
        _ACTIVE.reset(token)
        stop.set()
        worker.join()
        meter.sample()
