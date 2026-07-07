"""TabFM method.

TabFM (`google-research/tabfm`) is Google's zero-shot tabular foundation
model. It exposes a scikit-learn compatible API through the optional
``tabfm`` package.

Setup:
    pip install -U "tabfm[pytorch]"
"""

import inspect
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from TALENT.model.lib.data import (
    Dataset,
    data_label_process,
    data_nan_process,
)
from TALENT.model.methods.base import Method


class TabFMMethod(Method):
    """TabFM zero-shot classifier/regressor wrapper."""

    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert args.normalization == "none"
        assert args.cat_policy == "indices"
        assert args.num_policy == "none"
        assert args.tune is not True

    def data_format(self, is_train=True, N=None, C=None, y=None):
        if is_train:
            self.N, self.C, self.num_new_value, self.imputer, self.cat_new_value = data_nan_process(
                self.N, self.C, self.args.num_nan_policy, self.args.cat_nan_policy
            )
            self.y, self.y_info, self.label_encoder = data_label_process(
                self.y, self.is_regression
            )
            self.criterion = F.mse_loss if self.is_regression else F.cross_entropy
        else:
            N_test, C_test, _, _, _ = data_nan_process(
                N,
                C,
                self.args.num_nan_policy,
                self.args.cat_nan_policy,
                self.num_new_value,
                self.imputer,
                self.cat_new_value,
            )
            y_test, _, _ = data_label_process(
                y, self.is_regression, self.y_info, self.label_encoder
            )
            self.N_test = None if N_test is None else N_test["test"]
            self.C_test = None if C_test is None else C_test["test"]
            self.y_test = y_test["test"]

    def _to_frame(self, N_part, C_part):
        columns = []
        parts = []
        if N_part is not None:
            N_part = np.asarray(N_part)
            parts.append(N_part)
            columns.extend([f"num_{i}" for i in range(N_part.shape[1])])
        if C_part is not None:
            C_part = np.asarray(C_part).astype(str)
            parts.append(C_part)
            columns.extend([f"cat_{i}" for i in range(C_part.shape[1])])
        if not parts:
            raise ValueError("TabFM requires at least one feature column.")
        X = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        df = pd.DataFrame(X, columns=columns)
        for col in df.columns:
            if col.startswith("num_"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = df[col].astype("category")
        return df

    def _subsample_frame(self, X, y):
        sample_size = self.resolve_sample_size()
        if sample_size is None or len(X) <= sample_size:
            return X, y
        if not self.is_regression:
            from sklearn.model_selection import train_test_split

            X_sample, _, y_sample, _ = train_test_split(
                X,
                y,
                train_size=sample_size,
                stratify=y,
                random_state=self.args.seed,
            )
            return X_sample.reset_index(drop=True), y_sample
        rng = np.random.RandomState(self.args.seed)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        return X.iloc[idx].reset_index(drop=True), y[idx]

    def _load_tabfm_model(self):
        general = self.args.config.get("general", {}) or {}
        backend = general.get("backend", "pytorch")
        if backend != "pytorch":
            raise ValueError(
                "TALENT's TabFM wrapper currently supports backend='pytorch'. "
                "Install with: pip install -U \"tabfm[pytorch]\"."
            )
        try:
            from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
        except ImportError as e:
            raise ImportError(
                "TabFM requires the optional `tabfm` package with the PyTorch "
                "backend. Install it via: pip install -U \"tabfm[pytorch]\"."
            ) from e

        load_kwargs = {
            "model_type": "regression" if self.is_regression else "classification",
            "device": str(self.args.device),
            "use_cache": general.get("use_cache", True),
        }
        for key in ("checkpoint_path", "dtype"):
            if key in general:
                load_kwargs[key] = general[key]
        return tabfm_v1_0_0.load(**load_kwargs)

    def construct_model(self, model_config=None, cat_indices=None):
        try:
            from tabfm import TabFMClassifier, TabFMRegressor
        except ImportError as e:
            raise ImportError(
                "TabFM requires the optional `tabfm` package. Install it via: "
                "pip install -U \"tabfm[pytorch]\"."
            ) from e

        general = self.args.config.get("general", {}) or {}
        model = self._load_tabfm_model()
        target_cls = TabFMRegressor if self.is_regression else TabFMClassifier

        wrapper_kwargs = {"model": model, "random_state": self.args.seed}
        for key in (
            "n_estimators",
            "norm_methods",
            "feat_shuffle_method",
            "class_shift",
            "permute_categorical",
            "outlier_threshold",
            "max_num_features",
            "max_num_rows",
            "softmax_temperature",
            "average_logits",
            "use_amp",
            "batch_size",
            "verbose",
            "cat_encoder_mode",
            "binary_calibration_method",
            "multiclass_calibration_method",
            "num_folds_for_cv",
            "n_feature_crosses",
            "n_svd_features",
            "total_svd_pool",
            "enable_nnls",
            "nnls_beta",
            "calibration_lambda",
            "min_rows_for_single_val_split",
        ):
            if key in general:
                wrapper_kwargs[key] = general[key]

        accepted = set(inspect.signature(target_cls.__init__).parameters)
        wrapper_kwargs = {k: v for k, v in wrapper_kwargs.items() if k in accepted}
        if general.get("ensemble", False):
            self.model = target_cls.ensemble(model, **{
                k: v for k, v in wrapper_kwargs.items() if k != "model"
            })
        else:
            self.model = target_cls(**wrapper_kwargs)

    def fit(self, data, info, train=True, config=None):
        N, C, y = data
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = (
            self.D.is_binclass,
            self.D.is_multiclass,
            self.D.is_regression,
        )
        self.data_format(is_train=True)

        if not self.is_regression and self.y_info["n_classes"] > 10:
            raise ValueError(
                "TabFM v1.0.0 supports classification with at most 10 classes; "
                f"got {self.y_info['n_classes']}."
            )

        X_train = self._to_frame(
            None if self.N is None else self.N["train"],
            None if self.C is None else self.C["train"],
        )
        y_train = self.y["train"]
        X_train, y_train = self._subsample_frame(X_train, y_train)
        self.sampled_X = X_train
        self.sampled_Y = y_train
        self.construct_model()

        tic = time.time()
        self.model.fit(X_train, y_train)
        self.fit_time = time.time() - tic

    def predict(self, data, info, model_name):
        N, C, y = data
        self.data_format(False, N, C, y)
        X_test = self._to_frame(self.N_test, self.C_test)

        tic = time.time()
        if self.is_regression:
            test_logit = self.model.predict(X_test)
        else:
            test_logit = self.model.predict_proba(X_test)
        self.predict_time = time.time() - tic

        test_logit = np.asarray(test_logit).astype(np.float32)
        test_label = self.y_test
        if self.is_regression:
            t_pred = torch.tensor(test_logit).reshape(-1)
            t_lab = torch.tensor(test_label, dtype=torch.float32).reshape(-1)
            vl = self.criterion(t_pred, t_lab).item()
        else:
            vl = self.criterion(
                torch.tensor(test_logit), torch.tensor(test_label)
            ).item()
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)

        if self.is_regression and self.y_info.get("policy") == "mean_std":
            test_logit = test_logit * self.y_info["std"] + self.y_info["mean"]

        print("Test: loss={:.4f}".format(vl))
        for name, res in zip(metric_name, vres):
            print("[{}]={:.4f}".format(name, res))
        return vl, vres, metric_name, test_logit
