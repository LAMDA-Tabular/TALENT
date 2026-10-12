"""EXAONE-Tabular method.

EXAONE-Tabular (LG AI Research, 2026) is a pre-trained in-context tabular
foundation model for classification and regression. It takes plain NumPy
arrays with categorical features already encoded, which matches TALENT's
ordinal-encoded feature matrix. The released checkpoints are downloaded from
the Hugging Face Hub (``LG-AI-Research/EXAONE-Tabular``) by
``from_pretrained``.

The weights are released under the EXAONE AI Model License 1.2-NC
(research and education only).

Setup:
    pip install "exaonetabular @ git+https://github.com/LGAI-Research/EXAONE-Tabular.git"
"""

import numpy as np

from TALENT.model.methods.foundation_base import SklearnFoundationMethod, resolve_n_estimators


class EXAONETabularMethod(SklearnFoundationMethod):
    """EXAONE-Tabular (in-context tabular foundation model)."""

    def construct_model(self, model_config=None, cat_indices=None):
        try:
            from exaonetabular import EXAONETabularClassifier, EXAONETabularRegressor
        except ImportError as e:
            raise ImportError(
                "EXAONE-Tabular requires the `exaonetabular` package. Install it via: "
                "pip install \"exaonetabular @ git+https://github.com/LGAI-Research/EXAONE-Tabular.git\""
            ) from e

        general = self.args.config.get('general', {}) or {}
        # from_pretrained defaults to device="cpu", so pass TALENT's device.
        kwargs = dict(device=str(self.args.device), seed=self.args.seed)
        # `n_estimators` is TALENT's common key for EXAONE's `ensemble_count`.
        ensemble_count = resolve_n_estimators(self.args, None, 'ensemble_count')
        if ensemble_count is not None:
            kwargs['ensemble_count'] = ensemble_count
        for k in ('compute_dtype', 'max_vram_bytes', 'weights', 'revision'):
            if k in general:
                kwargs[k] = general[k]
        model_cls = EXAONETabularRegressor if self.is_regression else EXAONETabularClassifier
        self.model = model_cls.from_pretrained(**kwargs)

    # EXAONE-Tabular validates for dense 2-D float tables and 1-D NumPy targets
    # (a categorical-only dataset would otherwise arrive as an int64 matrix).
    def _fit_model(self, X, y):
        self.model.fit(np.asarray(X, dtype=np.float64), np.asarray(y).reshape(-1))

    def _predict_raw(self, X):
        return super()._predict_raw(np.asarray(X, dtype=np.float64))
