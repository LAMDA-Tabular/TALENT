"""Shared base class for external in-context-learning foundation models.

Most tabular foundation models ship a scikit-learn style estimator
(``fit`` / ``predict`` / ``predict_proba``) and need the raw features:
no normalization, categorical columns as ordinal indices, no numerical
encoding, and no hyper-parameter tuning. This base class implements that
shared pipeline once:

* ``data_format`` -- NaN handling, label processing and ordinal encoding of
  categorical columns (fitted on train, re-applied at test time);
* ``fit`` -- concatenates numerical and categorical columns (numerical first,
  so the categorical columns are ``cat_indices``), applies the row cap from
  :meth:`Method.subsample_train_rows`, builds the estimator via
  :meth:`construct_model` and fits it;
* ``predict`` -- calls ``predict`` (regression) or ``predict_proba``
  (classification), computes the loss and metrics, and returns regression
  predictions on the original target scale.

Subclasses only implement ``construct_model(model_config=None,
cat_indices=None)`` and may override :meth:`_fit_model` /
:meth:`_predict_raw` when the upstream API differs.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from TALENT.model.lib.data import (
    Dataset,
    data_nan_process,
    data_enc_process,
    data_label_process,
)
from TALENT.model.method_registry import weights_download_hint
from TALENT.model.methods.base import Method
from TALENT.model.utils import probability_nll


def resolve_n_estimators(args, default, *legacy_keys):
    """Ensemble size from ``config['general']``.

    ``n_estimators`` is the common key for every foundation model; the
    model-specific ``legacy_keys`` (e.g. ``n_ensembles`` for TabDPT) are still
    accepted for backward compatibility.
    """
    general = args.config.get('general', {}) or {}
    for key in ('n_estimators',) + legacy_keys:
        if key in general:
            return general[key]
    return default


class SklearnFoundationMethod(Method):
    """Base class for pre-trained, sklearn-style tabular foundation models."""

    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert args.normalization == 'none'
        assert args.cat_policy == 'indices'
        assert args.num_policy == 'none'
        assert args.tune is not True

    def weights_hint(self):
        """Context manager that explains how to obtain the weights if loading fails."""
        return weights_download_hint(getattr(self.args, 'model_type', type(self).__name__))

    def data_format(self, is_train=True, N=None, C=None, y=None):
        if is_train:
            self.N, self.C, self.num_new_value, self.imputer, self.cat_new_value = data_nan_process(
                self.N, self.C, self.args.num_nan_policy, self.args.cat_nan_policy
            )
            self.y, self.y_info, self.label_encoder = data_label_process(self.y, self.is_regression)
            self.N, self.C, self.ord_encoder, self.mode_values, self.cat_encoder = data_enc_process(
                self.N, self.C, self.args.cat_policy
            )
            self.criterion = F.cross_entropy if not self.is_regression else F.mse_loss
        else:
            N_test, C_test, _, _, _ = data_nan_process(
                N, C, self.args.num_nan_policy, self.args.cat_nan_policy,
                self.num_new_value, self.imputer, self.cat_new_value
            )
            N_test, C_test, _, _, _ = data_enc_process(
                N_test, C_test, self.args.cat_policy, None,
                self.ord_encoder, self.mode_values, self.cat_encoder
            )
            y_test, _, _ = data_label_process(y, self.is_regression, self.y_info, self.label_encoder)
            if N_test is not None and C_test is not None:
                self.N_test, self.C_test = N_test['test'], C_test['test']
            elif N_test is None and C_test is not None:
                self.N_test, self.C_test = None, C_test['test']
            else:
                self.N_test, self.C_test = N_test['test'], None
            self.y_test = y_test['test']

    def construct_model(self, model_config=None, cat_indices=None):
        raise NotImplementedError

    def _fit_model(self, X, y):
        """Fit the upstream estimator (override if its API differs)."""
        self.model.fit(X, y)

    def _predict_raw(self, X):
        """Regression predictions or ``(N, K)`` class probabilities."""
        if self.is_regression:
            return self.model.predict(X)
        return self.model.predict_proba(X)

    def _as_categorical_frame(self, X):
        """Return ``X`` as a DataFrame whose categorical columns (``self.cat_indices``)
        have pandas ``category`` dtype, for estimators that detect categorical
        features from column dtypes instead of taking explicit indices."""
        import pandas as pd
        frame = pd.DataFrame(X)
        for i in self.cat_indices:
            frame[i] = pd.Categorical(frame[i].astype(np.int64))
        return frame

    def fit(self, data, info, train=True, config=None):
        N, C, y = data
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = (
            self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        )
        self.data_format(is_train=True)

        sampled_Y = self.y['train']
        cat_indices = []
        if self.N is not None and self.C is not None:
            sampled_X = np.concatenate((self.N['train'], self.C['train']), axis=1)
            n_num = self.N['train'].shape[1]
            cat_indices = list(range(n_num, n_num + self.C['train'].shape[1]))
        elif self.N is None and self.C is not None:
            sampled_X = self.C['train']
            cat_indices = list(range(self.C['train'].shape[1]))
        else:
            sampled_X = self.N['train']

        # Row cap: config['general']['sample_size'] override, else the
        # registry's train_row_limit.
        sampled_X, sampled_Y = self.subsample_train_rows(sampled_X, sampled_Y)

        self.sampled_X = sampled_X
        self.sampled_Y = sampled_Y
        self.cat_indices = cat_indices
        # Weights are downloaded either when the estimator is built or on its
        # first fit, depending on the upstream package.
        with self.weights_hint():
            self.construct_model(cat_indices=cat_indices)
            tic = time.time()
            self._fit_model(sampled_X, sampled_Y)
            self.fit_time = time.time() - tic

    def predict(self, data, info, model_name):
        N, C, y = data
        self.data_format(False, N, C, y)
        if self.N_test is not None and self.C_test is not None:
            Test_X = np.concatenate((self.N_test, self.C_test), axis=1)
        elif self.N_test is None and self.C_test is not None:
            Test_X = self.C_test
        else:
            Test_X = self.N_test

        tic = time.time()
        test_logit = np.asarray(self._predict_raw(Test_X))
        self.predict_time = time.time() - tic

        test_label = self.y_test
        if self.is_regression:
            t_pred = torch.tensor(test_logit, dtype=torch.float32).reshape(-1)
            t_lab = torch.tensor(test_label, dtype=torch.float32).reshape(-1)
            vl = self.criterion(t_pred, t_lab).item()
        else:
            # predict_proba returns probabilities, so the loss is their NLL.
            vl = probability_nll(test_logit, test_label)
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)

        # Denormalize regression predictions for the returned value
        if self.is_regression and self.y_info.get('policy') == 'mean_std':
            test_logit = test_logit * self.y_info['std'] + self.y_info['mean']
        print('Test: loss={:.4f}'.format(vl))
        for name, res in zip(metric_name, vres):
            print('[{}]={:.4f}'.format(name, res))
        return vl, vres, metric_name, test_logit
