"""Causilo method.

Causilo (Nums AI, September 2026, arXiv 2609.22866) is a pre-trained
in-context tabular foundation model for classification and regression with a
scikit-learn interface. More than ten classes are handled natively through
error-correcting output codes.

Causilo detects categorical features from column dtypes (it has no
categorical-index argument), so the wrapper passes a DataFrame whose
categorical columns have pandas ``category`` dtype.

The model weights are released under the Causilo License v1.0, which permits
non-commercial research and benchmarking; commercial use requires a separate
license.

Setup:
    pip install -U causilo        # requires PyTorch >= 2.13
"""

from TALENT.model.methods.foundation_base import SklearnFoundationMethod


class CausiloMethod(SklearnFoundationMethod):
    """Causilo (in-context tabular foundation model)."""

    def construct_model(self, model_config=None, cat_indices=None):
        try:
            from causilo import CausiloClassifier, CausiloRegressor
        except ImportError as e:
            raise ImportError(
                "Causilo requires the `causilo` package (which needs PyTorch >= 2.13). "
                "Install it via: pip install -U causilo"
            ) from e

        general = self.args.config.get('general', {}) or {}
        kwargs = dict(
            n_estimators=general.get('n_estimators', 8),
            random_state=self.args.seed,
            device=str(self.args.device),
        )
        for k in ('use_kv_cache', 'retain_preprocessing'):
            if k in general:
                kwargs[k] = general[k]
        model_cls = CausiloRegressor if self.is_regression else CausiloClassifier
        self.model = model_cls(**kwargs)

    def _fit_model(self, X, y):
        self.model.fit(self._as_categorical_frame(X), y)

    def _predict_raw(self, X):
        return super()._predict_raw(self._as_categorical_frame(X))
