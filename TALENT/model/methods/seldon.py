"""Local Seldon and Seldon-mini inference using Neuralk's neuralk-oss package."""

from pathlib import Path

from TALENT.model.methods.foundation_base import SklearnFoundationMethod


class SeldonMethod(SklearnFoundationMethod):
    """Load the task-specific classifier or regressor and its publisher recipe."""

    MODEL_NAME = "seldon"

    def construct_model(self, model_config=None, cat_indices=None):
        try:
            from neuralk_oss import SeldonClassifier, SeldonRegressor
        except ImportError as exc:
            raise ImportError(
                "Local Seldon inference requires pip install 'neuralk-oss>=0.3.0'."
            ) from exc

        general = self.args.config.get("general", {}) or {}
        checkpoint = self.MODEL_NAME
        if general.get("model_path") is not None:
            checkpoint = Path(general["model_path"]).expanduser()
            if not checkpoint.is_dir():
                raise FileNotFoundError("Seldon's model_path must name an existing local checkpoint directory.")
        estimator = SeldonRegressor if self.is_regression else SeldonClassifier
        options = {"device": str(self.args.device), "random_state": self.args.seed}
        # Omitted options retain the publisher's task- and size-specific recipe.
        for key in ("n_estimators", "revision", "amp", "max_context_size", "query_chunk_size"):
            if key in general:
                options[key] = general[key]
        self.model = estimator.from_pretrained(checkpoint, **options)
        task = "regressor" if self.is_regression else "classifier"
        self.resolved_checkpoint = (
            str(checkpoint) if isinstance(checkpoint, Path)
            else f"Neuralk-AI/{self.MODEL_NAME}/{self.MODEL_NAME}-{task}.safetensors"
        )

    def _fit_model(self, X, y):
        self.model.fit(self._as_categorical_frame(X), y)

    def _predict_raw(self, X):
        return super()._predict_raw(self._as_categorical_frame(X))


class SeldonMiniMethod(SeldonMethod):
    MODEL_NAME = "seldon-mini"
