"""Xiaomi-TabLDM method.

Xiaomi-TabLDM (Xiaomi, September 2026, arXiv 2609.03880) is a pre-trained
in-context tabular foundation model for classification and regression with a
scikit-learn interface; checkpoints are downloaded automatically from the
Hugging Face Hub (``occams/Xiaomi-TabLDM``). Upstream defaults are used unless
overridden in ``config['general']``. Each option is only forwarded if the
installed estimator accepts it.

The code is Apache-2.0; the README states that the weights are released under
the Xiaomi-TabLDM Non-Commercial License -- check the model card before
commercial use.

Setup:
    pip install git+https://github.com/xiaomi-research/xiaomi-tabldm.git
"""

import inspect

from TALENT.model.methods.foundation_base import SklearnFoundationMethod


class TabLDMMethod(SklearnFoundationMethod):
    """Xiaomi-TabLDM (in-context tabular foundation model)."""

    _OPTIONAL_KWARGS = (
        'n_estimators', 'norm_methods', 'feat_shuffle_method', 'max_num_features',
        'outlier_threshold', 'softmax_temperature', 'average_logits', 'support_many_classes',
        'batch_size', 'kv_cache', 'model_path', 'checkpoint_version', 'use_amp', 'offload_mode',
        'verbose', 'enhance_candidates', 'validation', 'k_fold', 'n_splits',
    )

    def construct_model(self, model_config=None, cat_indices=None):
        try:
            from tabldm import TabLDMClassifier, TabLDMRegressor
        except ImportError as e:
            raise ImportError(
                "Xiaomi-TabLDM requires the `tabldm` package. Install it via: "
                "pip install git+https://github.com/xiaomi-research/xiaomi-tabldm.git"
            ) from e

        general = self.args.config.get('general', {}) or {}
        kwargs = dict(device=str(self.args.device), random_state=self.args.seed)
        if not self.is_regression and cat_indices:
            kwargs['categorical_indices'] = list(cat_indices)
        for k in self._OPTIONAL_KWARGS:
            if k in general:
                kwargs[k] = general[k]

        model_cls = TabLDMRegressor if self.is_regression else TabLDMClassifier
        accepted = set(inspect.signature(model_cls.__init__).parameters)
        self.model = model_cls(**{k: v for k, v in kwargs.items() if k in accepted})
