"""TabPFN v2.5 method.

TabPFN v2.5 (PriorLabs, November 2025) is the intermediate release between
v2 (Nature 2025) and v3 (2026). It scales in-context learning to ~50 000
rows × 2 000 features and is a drop-in replacement for v2 with better
accuracy and speed.

Like v3, v2.5 is distributed through the upstream `tabpfn>=8.0.0`
package; we pin the version via ``ModelVersion.V2_5`` when the version-
aware ``create_default_for_version`` classmethod is available, with a
plain constructor fallback otherwise.

Setup:
    pip install -U 'tabpfn>=8.0.0'
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


class TabPFNv2_5Method(Method):
    """TabPFN v2.5 (in-context tabular foundation model).

    Uses the official `tabpfn>=8.0.0` package. Pins the model version to
    v2.5 via ``ModelVersion.V2_5`` when available; falls back to whatever
    the installed package's default model version is otherwise.
    """

    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
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
        cat_indices = cat_indices or []
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
        except ImportError as e:
            raise ImportError(
                "TabPFN v2.5 requires the `tabpfn` package (>=8.0.0). "
                "Install it via: pip install -U 'tabpfn>=8.0.0'."
            ) from e

        general = self.args.config.get('general', {}) or {}
        n_est_default = 8 if self.is_regression else 4

        kwargs = dict(
            device=self.args.device,
            random_state=self.args.seed,
            ignore_pretraining_limits=general.get('ignore_pretraining_limits', True),
            categorical_features_indices=cat_indices,
            n_estimators=general.get('n_estimators', n_est_default),
        )
        # Optional user-supplied knobs forwarded if present.
        for k in ('softmax_temperature', 'memory_saving_mode', 'fit_mode',
                  'inference_precision', 'model_path'):
            if k in general:
                kwargs[k] = general[k]

        model_cls = TabPFNRegressor if self.is_regression else TabPFNClassifier

        # Prefer the version-pinning classmethod when available so we are
        # explicit about wanting v2.5 even on systems that default to v3.
        try:
            from tabpfn.constants import ModelVersion
            version = getattr(ModelVersion, 'V2_5', None)
            if version is not None and hasattr(model_cls, 'create_default_for_version'):
                self.model = model_cls.create_default_for_version(version, **kwargs)
                return
        except ImportError:
            pass

        # Fallback path -- use the installed package's default model version.
        self.model = model_cls(**kwargs)

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

        # Optional sample_size cap (v2.5 supports up to ~50k natively).
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
        if self.is_regression:
            test_logit = self.model.predict(Test_X)
        else:
            test_logit = self.model.predict_proba(Test_X)
        self.predict_time = time.time() - tic

        test_label = self.y_test
        if self.is_regression:
            t_pred = torch.tensor(test_logit, dtype=torch.float32).reshape(-1)
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
