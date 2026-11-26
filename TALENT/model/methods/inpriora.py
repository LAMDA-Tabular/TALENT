"""Inpriora classifier and regressor using the publisher's inference package."""

from inspect import signature

from sklearn.utils.validation import check_array

from TALENT.model.methods.foundation_base import SklearnFoundationMethod


class InprioraMethod(SklearnFoundationMethod):
    def construct_model(self, model_config=None, cat_indices=None):
        # The publisher's encoder uses this keyword even though its package
        # metadata currently allows older scikit-learn versions.
        if "ensure_all_finite" not in signature(check_array).parameters:
            raise ImportError("Inpriora requires scikit-learn>=1.6,<2 for its feature encoder.")
        try:
            from inpriora import InprioraClassifier, InprioraRegressor
        except ImportError as exc:
            raise ImportError(
                "Inpriora requires pip install git+https://github.com/ai-for-data/inpriora.git"
            ) from exc

        general = self.args.config.get("general", {}) or {}
        options = {"device": str(self.args.device), "random_seed": self.args.seed}
        for key in ("n_estimators", "cache", "temperature", "amp", "model_path", "revision", "offline"):
            if key in general:
                options[key] = general[key]
        estimator = InprioraRegressor if self.is_regression else InprioraClassifier
        self.model = estimator(**options)

    def _fit_model(self, X, y):
        self.model.fit(self._as_categorical_frame(X), y)
        # The fitted publisher estimator records the actual local artifact,
        # including its Hub snapshot directory for downloaded weights.
        self.resolved_checkpoint = self.model.model_paths_["weights_path"]

    def _predict_raw(self, X):
        return super()._predict_raw(self._as_categorical_frame(X))
