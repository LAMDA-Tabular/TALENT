"""Batch prediction for adapters whose queries share a fixed training context."""

import gc
import logging
from numbers import Integral

import numpy as np
import torch


def query_batch_size(general, default=None):
    """Resolve the common query-batch setting, using the adapter default for zero."""
    size = general.get("predict_chunk_size")
    if size is None:
        return default
    if isinstance(size, bool) or not isinstance(size, Integral) or size < 0:
        raise ValueError("predict_chunk_size must be a nonnegative integer or None")
    return int(size) or default


def predict_in_batches(predict, n_rows, batch_size=None, *, retry_cuda_oom=False):
    """Score ordered row slices and collect all outputs before calculating metrics."""
    size = min(batch_size or n_rows, n_rows)
    if n_rows == 0:
        return predict(0, 0)
    start = 0
    parts = []
    while start < n_rows:
        stop = min(start + size, n_rows)
        try:
            prediction = predict(start, stop)
        except RuntimeError as error:
            message = str(error).lower()
            cuda_oom = isinstance(error, torch.cuda.OutOfMemoryError) or (
                "cuda" in message and "out of memory" in message
            )
            if not retry_cuda_oom or not cuda_oom or stop - start <= 1:
                raise
        else:
            if stop - start == n_rows:
                return prediction
            parts.append(np.asarray(prediction))
            start = stop
            continue

        # Release the failed call's traceback before retrying the same rows.
        size = (stop - start) // 2
        gc.collect()
        torch.cuda.empty_cache()
        logging.getLogger(__name__).warning("CUDA memory exhausted; retrying with %d query rows", size)
    return np.concatenate(parts, axis=0)
