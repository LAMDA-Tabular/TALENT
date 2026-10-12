"""NVIDIA Kumo Tabular method.

Kumo Tabular (NVIDIA, September 2026) is an in-context tabular foundation
model for classification and regression, released in three sizes (small ~28M,
medium ~62M, large ~215M parameters) under the OpenMDW-1.1 license, which
permits commercial use. More than ten classes are handled through
error-correcting output codes; regression predicts 999 quantiles, whose mean
is used as the point prediction.

This wrapper follows NVIDIA's own TALENT adapter
(``benchmark/tabular/talent/models.py`` in ``NVIDIA/structured-data-models``):
numerical NaNs are passed to the model natively instead of being imputed,
categorical values are passed as categorical columns, low-cardinality
numerical columns are typed as categorical from the training rows, and
inference runs under fp16 autocast on CUDA with 8 (small/medium) or 16 (large)
estimators. ``config['general']`` selects ``size``, ``num_estimators`` and
``low_cardinality``.

Setup (Python >= 3.11):
    pip install git+https://github.com/NVIDIA/structured-data-models.git
"""

import time
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from TALENT.model.lib.data import (
    Dataset,
    data_label_process,
    data_nan_process,
)
from TALENT.model.method_registry import weights_download_hint
from TALENT.model.methods.base import Method
from TALENT.model.methods.foundation_base import resolve_n_estimators
from TALENT.model.utils import probability_nll

_DEFAULT_NUM_ESTIMATORS = {"small": 8, "medium": 8, "large": 16}


@lru_cache(maxsize=2)
def _load_kumo_tabular(task, size, device):
    """Load (and cache) the pretrained model; each fit() replaces the context."""
    try:
        import sdm
    except ImportError as e:
        raise ImportError(
            "Kumo Tabular requires NVIDIA's `structured-data-models` package (Python >= 3.11). "
            "Install it via: pip install git+https://github.com/NVIDIA/structured-data-models.git"
        ) from e
    return sdm.models.KumoTabular(task=task, size=size, device=device)


def _as_float(arrays):
    """Float view of the numerical splits, keeping NaNs (no imputation)."""
    if arrays is None:
        return None
    return {
        split: np.asarray(values, dtype=float).reshape(len(values), -1)
        for split, values in arrays.items()
    }


def _target_vector(values):
    values = np.asarray(values)
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    return values


class KumoTabularMethod(Method):
    """NVIDIA Kumo Tabular (in-context tabular foundation model)."""

    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert args.normalization == 'none'
        assert args.cat_policy == 'indices'
        assert args.num_policy == 'none'
        assert args.tune is not True

        general = self.args.config.get('general', {}) or {}
        self._size = general.get('size', 'large')
        if self._size not in _DEFAULT_NUM_ESTIMATORS:
            raise ValueError(
                f"Kumo Tabular size must be one of {sorted(_DEFAULT_NUM_ESTIMATORS)}, got {self._size!r}."
            )
        # `n_estimators` is TALENT's common key (`num_estimators` is the upstream name).
        self._num_estimators = resolve_n_estimators(
            self.args, _DEFAULT_NUM_ESTIMATORS[self._size], 'num_estimators'
        )
        self._low_cardinality = general.get('low_cardinality', 'infer')
        self._device = torch.device(self.args.device)
        self._numerical_stypes = []

    def data_format(self, is_train=True, N=None, C=None, y=None):
        # Missing numerical values reach the model as NaN, which Kumo Tabular
        # handles natively, instead of TALENT's train-mean imputation.
        if is_train:
            _, self.C, self.num_new_value, self.imputer, self.cat_new_value = data_nan_process(
                None, self.C, self.args.num_nan_policy, self.args.cat_nan_policy
            )
            self.N = _as_float(self.N)
            self.y, self.y_info, self.label_encoder = data_label_process(self.y, self.is_regression)
            self.y = {split: _target_vector(values) for split, values in self.y.items()}
            self.criterion = F.mse_loss if self.is_regression else F.nll_loss
        else:
            _, C_test, _, _, _ = data_nan_process(
                None, C, self.args.num_nan_policy, self.args.cat_nan_policy,
                self.num_new_value, self.imputer, self.cat_new_value
            )
            N_test = _as_float(N)
            y_test, _, _ = data_label_process(y, self.is_regression, self.y_info, self.label_encoder)
            self.N_test = None if N_test is None else N_test['test']
            self.C_test = None if C_test is None else C_test['test']
            self.y_test = _target_vector(y_test['test'])

    def _infer_numerical_stypes(self, numerical):
        if numerical is None:
            return []
        if self._low_cardinality == 'off':
            return ['numerical'] * numerical.shape[1]
        # Typed from the train rows: numerical columns with 2 or 3 distinct
        # values become categorical (as in NVIDIA's TALENT/TabArena adapters).
        import sdm
        frame = pd.DataFrame(numerical).rename(columns=str)
        stypes = sdm.infer_stypes(frame, _low_cardinality=self._low_cardinality)
        return [str(stypes[column]) for column in frame.columns]

    def _to_table(self, numerical, categorical):
        import sdm
        data, stypes = {}, {}
        if numerical is not None:
            for i, column in enumerate(np.asarray(numerical).T):
                data[f"num_{i}"] = column
                stypes[f"num_{i}"] = self._numerical_stypes[i]
        if categorical is not None:
            for i, column in enumerate(np.asarray(categorical).T):
                data[f"cat_{i}"] = column
                stypes[f"cat_{i}"] = "categorical"
        return sdm.TableTensor.from_columns(data=data, stypes=stypes, device=self._device)

    def _to_target(self, values):
        import sdm
        return sdm.TableTensor.from_columns(
            data={"target": values},
            stypes={"target": "numerical" if self.is_regression else "categorical"},
            device=self._device,
        )

    def construct_model(self, model_config=None):
        task = "regression" if self.is_regression else "classification"
        self.model = _load_kumo_tabular(task, self._size, self._device)

    def fit(self, data, info, train=True, config=None):
        N, C, y = data
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = (
            self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        )
        self.data_format(is_train=True)

        # Optional row cap via config['general']['sample_size'] (no registry
        # limit). Subsample row indices so N, C and y stay aligned.
        n_train = len(self.y['train'])
        idx, y_train = self.subsample_train_rows(np.arange(n_train).reshape(-1, 1), self.y['train'])
        idx = idx.reshape(-1)
        N_train = None if self.N is None else self.N['train'][idx]
        C_train = None if self.C is None else self.C['train'][idx]

        self._numerical_stypes = self._infer_numerical_stypes(N_train)
        x_train = self._to_table(N_train, C_train)
        y_train = self._to_target(y_train)
        with weights_download_hint(getattr(self.args, 'model_type', 'kumo_tabular')):
            self.construct_model()
        generator = torch.Generator(device=self._device).manual_seed(self.args.seed)

        tic = time.time()
        with torch.amp.autocast(self._device.type, torch.float16, enabled=x_train.is_cuda):
            self.model.fit(
                x=x_train, y=y_train, num_estimators=self._num_estimators, generator=generator,
            )
        if x_train.is_cuda:
            torch.cuda.synchronize(x_train.device)
        self.fit_time = time.time() - tic

    def predict(self, data, info, model_name):
        N, C, y = data
        self.data_format(False, N, C, y)
        x_test = self._to_table(self.N_test, self.C_test)

        tic = time.time()
        with torch.amp.autocast(self._device.type, torch.float16, enabled=x_test.is_cuda):
            out = self.model.predict(x_test)
        if x_test.is_cuda:
            torch.cuda.synchronize(x_test.device)
        self.predict_time = time.time() - tic

        if self.is_regression:
            # Point prediction = mean of the 999 predicted quantiles.
            test_logit = out.numerical.float().mean(dim=-1).cpu().numpy()
            vl = self.criterion(
                torch.as_tensor(test_logit).reshape(-1),
                torch.as_tensor(self.y_test, dtype=torch.float32).reshape(-1),
            ).item()
        else:
            # Probability columns are named after the class values.
            columns = [str(value) for value in self.y_info['classes']]
            test_logit = out.to_pandas()[columns].to_numpy()
            vl = probability_nll(test_logit, self.y_test)
        vres, metric_name = self.metric(test_logit, self.y_test, self.y_info)

        # Denormalize regression predictions for the returned value
        if self.is_regression and self.y_info.get('policy') == 'mean_std':
            test_logit = test_logit * self.y_info['std'] + self.y_info['mean']
        print('Test: loss={:.4f}'.format(vl))
        for name, res in zip(metric_name, vres):
            print('[{}]={:.4f}'.format(name, res))
        return vl, vres, metric_name, test_logit
