"""TabDPT method.

TabDPT (`Layer 6 AI <https://github.com/layer6ai-labs/TabDPT-inference>`_)
is a tabular foundation model that combines in-context learning with
retrieval and self-supervised pre-training on real data. Unlike TabPFN /
TabICL it does not have a hard context-size limit -- the retrieval step
selects the most relevant rows from the training set for each query.

The wrapper uses the upstream ``tabdpt`` package (sklearn-compatible
API: ``TabDPTClassifier`` / ``TabDPTRegressor``) and exposes the runtime
parameters (``n_ensembles``, ``context_size``, ``temperature``) through
``config['general']`` so they can be tuned without code changes.

Setup:
    pip install -U tabdpt
"""

from TALENT.model.methods.base import Method
import torch
import numpy as np
import torch.nn.functional as F

from TALENT.model.lib.data import (
    Dataset,
    data_nan_process,
    data_enc_process,
    data_label_process,
)
import time


class TabDPTMethod(Method):
    """TabDPT (tabular foundation model with retrieval + ICL)."""

    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        # TabDPT does its own internal preprocessing, so we keep the
        # outer pipeline minimal -- same convention as tabpfn_v2 /
        # tabicl / mitra / limix.
        assert args.normalization == 'none'
        assert args.cat_policy == 'indices'
        assert args.num_policy == 'none'
        assert args.tune is not True

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
        try:
            from tabdpt import TabDPTClassifier, TabDPTRegressor
        except ImportError as e:
            raise ImportError(
                "TabDPT requires the `tabdpt` package. "
                "Install it via: pip install -U tabdpt"
            ) from e

        # TabDPT's constructors take few arguments at __init__ time --
        # most runtime knobs (n_ensembles, context_size, temperature, ...)
        # are passed to predict(). Persist them on self for later.
        general = self.args.config.get('general', {}) or {}
        self._predict_kwargs = {
            "n_ensembles": general.get("n_ensembles", 8 if not self.is_regression else 2),
            "context_size": general.get("context_size", 2048),
            "seed": general.get("seed", self.args.seed),
        }
        if not self.is_regression:
            # Classification-only knobs
            self._predict_kwargs.setdefault("temperature", general.get("temperature", 0.8))
            self._predict_kwargs.setdefault(
                "permute_classes", general.get("permute_classes", True)
            )

        if self.is_regression:
            self.model = TabDPTRegressor()
        else:
            self.model = TabDPTClassifier()

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

        # Optional sample_size cap. TabDPT scales via retrieval so it can
        # generally handle large train sets, but the cap is exposed for
        # benchmarking / time-budget consistency with other methods.
        general = self.args.config.get('general', {}) or {}
        sample_size = general.get('sample_size', None)
        if sample_size is not None and sampled_X.shape[0] > sample_size:
            if not self.is_regression:
                from sklearn.model_selection import train_test_split
                sampled_X, _, sampled_Y, _ = train_test_split(
                    sampled_X, sampled_Y,
                    train_size=sample_size,
                    stratify=sampled_Y,
                    random_state=self.args.seed,
                )
            else:
                rng = np.random.RandomState(self.args.seed)
                idx = rng.choice(sampled_X.shape[0], size=sample_size, replace=False)
                sampled_X = sampled_X[idx]
                sampled_Y = sampled_Y[idx]

        self.sampled_X = sampled_X
        self.sampled_Y = sampled_Y
        self.construct_model(cat_indices=cat_indices)

        tic = time.time()
        self.model.fit(sampled_X, sampled_Y)
        self.fit_time = time.time() - tic

    def _predict_safely(self, X):
        """Call ``model.predict`` / ``predict_proba`` with the runtime knobs
        we cached at construct time. Falls back gracefully if the upstream
        signature does not accept one of our kwargs.
        """
        if self.is_regression:
            try:
                return self.model.predict(X, **self._predict_kwargs)
            except TypeError:
                return self.model.predict(X)
        # Classification -- prefer predict_proba for calibrated probabilities.
        if hasattr(self.model, "predict_proba"):
            try:
                return self.model.predict_proba(X, **self._predict_kwargs)
            except TypeError:
                try:
                    return self.model.predict_proba(X)
                except Exception:
                    pass
        # Fallback: predict() returns class labels; manufacture one-hot.
        try:
            preds = self.model.predict(X, **self._predict_kwargs)
        except TypeError:
            preds = self.model.predict(X)
        preds = np.asarray(preds).astype(int)
        n_classes = int(self.y_info.get('n_classes', 2))
        one_hot = np.zeros((len(preds), n_classes), dtype=np.float32)
        one_hot[np.arange(len(preds)), preds] = 1.0
        return one_hot

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
        test_logit = self._predict_safely(Test_X)
        self.predict_time = time.time() - tic

        test_logit = np.asarray(test_logit).astype(np.float32)
        test_label = self.y_test
        if self.is_regression:
            t_pred = torch.tensor(test_logit).reshape(-1)
            t_lab = torch.tensor(test_label, dtype=torch.float32).reshape(-1)
            vl = self.criterion(t_pred, t_lab).item()
        else:
            vl = self.criterion(torch.tensor(test_logit), torch.tensor(test_label)).item()
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)

        # Denormalize regression predictions for the returned value
        if self.is_regression and self.y_info.get('policy') == 'mean_std':
            test_logit = test_logit * self.y_info['std'] + self.y_info['mean']

        print('Test: loss={:.4f}'.format(vl))
        for name, res in zip(metric_name, vres):
            print('[{}]={:.4f}'.format(name, res))
        return vl, vres, metric_name, test_logit
