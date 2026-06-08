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


def check_softmax(logits):
    """Convert raw logits to probabilities if not already in [0, 1] summing to 1."""
    if np.any((logits < 0) | (logits > 1)) or (not np.allclose(logits.sum(axis=-1), 1, atol=1e-5)):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    return logits


class TabICLv2Method(Method):
    """TabICL v2 method — supports both classification AND regression.

    Uses the official `tabicl>=2.0.0` package which adds regression support
    via `TabICLRegressor` (v1 was classification-only).

    Setup:
        pip install -U 'tabicl>=2.0.0'
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
        import inspect
        cat_indices = cat_indices or []
        try:
            from tabicl import TabICLClassifier, TabICLRegressor
        except ImportError as e:
            raise ImportError(
                "TabICL v2 requires the `tabicl` package (>=2.0.0). "
                "Install it via: pip install -U 'tabicl>=2.0.0'."
            ) from e

        general = self.args.config.get('general', {}) or {}
        common = dict(
            device=self.args.device,
            random_state=self.args.seed,
            n_estimators=general.get('n_estimators', 32),
            batch_size=general.get('batch_size', 8),
            use_amp=general.get('use_amp', True),
            allow_auto_download=general.get('allow_auto_download', True),
            verbose=general.get('verbose', False),
        )
        # checkpoint_version / model_path are optional — let users pin a specific ckpt.
        for k in ('checkpoint_version', 'model_path', 'norm_methods',
                  'feat_shuffle_method', 'outlier_threshold', 'inference_config'):
            if k in general:
                common[k] = general[k]

        target_cls = TabICLRegressor if self.is_regression else TabICLClassifier
        if not self.is_regression:
            # Classifier-only knobs
            common.update(
                softmax_temperature=general.get('softmax_temperature', 0.9),
                average_logits=general.get('average_logits', True),
                use_hierarchical=general.get('use_hierarchical', True),
                class_shift=general.get('class_shift', True),
            )
        accepted = set(inspect.signature(target_cls.__init__).parameters)
        kwargs = {k: v for k, v in common.items() if k in accepted}
        self.model = target_cls(**kwargs)

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

        # Optional sample_size cap, since TabICL keeps all training rows in-context.
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
        self.model.fit(self.sampled_X, self.sampled_Y)
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
