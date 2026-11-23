"""LimiX-2 method.

LimiX-2 (StableAI / Tsinghua, September 2026, arXiv 2609.17488) is a 400M
parameter structured-data foundation model; one pretrained model performs
classification (2-10 classes) and regression in a single forward pass.

LimiX-2 uses a new (v2.0) architecture that TALENT's bundled LimiX-16M code
cannot load, so the wrapper uses the upstream ``LimiX`` package, whose
``LimiXPredictor`` factory routes the checkpoint to the right implementation.
The checkpoint is used from ``TALENT/model/models/models_limix/LimiX-2.ckpt``
when bundled, and otherwise downloaded automatically from the Hugging Face Hub
(``stable-ai/LimiX-2``). The weights are released under the StableAI LimiX
Non-Commercial License.

Setup (Python >= 3.12, PyTorch >= 2.9.1):
    pip install git+https://github.com/limix-ldm-ai/LimiX.git
"""

from importlib import import_module
from pathlib import Path

from TALENT.model.methods.limix import LimiXMethod

_INSTALL = "pip install git+https://github.com/limix-ldm-ai/LimiX.git (Python >= 3.12)"


def _predictor_module():
    try:
        return import_module("ldm_limix.inference.predictor")
    except ModuleNotFoundError as exc:
        if exc.name != "ldm_limix":
            raise
    try:
        return import_module("inference.predictor")
    except ModuleNotFoundError as exc:
        if exc.name not in ("inference", "inference.predictor"):
            raise
        raise ImportError(f"LimiX-2 requires the publisher's inference package: {_INSTALL}") from exc


class LimiXV2Method(LimiXMethod):
    """LimiX-2 (in-context tabular foundation model, upstream ``LimiX`` package)."""

    CHECKPOINT = "LimiX-2.ckpt"
    HF_REPO = "stable-ai/LimiX-2"
    MAX_CLASSES = 10
    USE_ORIGINAL_CONFIG = True

    def _inference_config_path(self, module=None):
        """``config['general']['inference_config']`` if set, else the default
        no-retrieval v2 config shipped with the upstream package."""
        general = self.args.config.get('general', {}) or {}
        if general.get('inference_config'):
            return general['inference_config']
        if module is None:
            module = _predictor_module()
        name = ("reg" if self.is_regression else "cls") + "_default_noretrieval_v2.json"
        path = Path(module.__file__).resolve().parents[1] / "config" / name
        # After 1016 the publisher retained the original recipe under this spelling.
        original = path.with_name(path.stem + "_lagacy.json")
        if self.USE_ORIGINAL_CONFIG and original.is_file():
            path = original
        if not path.is_file():
            raise FileNotFoundError(
                f"LimiX-2 inference config {name!r} not found next to the installed `inference` "
                f"package ({path}). Reinstall LimiX ({_INSTALL}) or set "
                "config['general']['inference_config'] to the JSON file."
            )
        return str(path)

    def construct_model(self, model_config = None):
        # Checked before the 400M-parameter checkpoint is loaded.
        if not self.is_regression and not 2 <= self.n_classes <= self.MAX_CLASSES:
            raise ValueError(
                f"LimiX-2 supports classification with 2 to {self.MAX_CLASSES} classes; "
                f"got {self.n_classes}."
            )
        module = _predictor_module()
        if not self.USE_ORIGINAL_CONFIG and module.__name__ == "inference.predictor":
            raise ImportError("LimiX-2-1016 requires the release with the ldm_limix package and 1016 configs.")
        general = self.args.config.get('general', {}) or {}
        self.resolved_checkpoint = self.checkpoint_path()
        kwargs = dict(
            device = self.args.device,
            model_path = self.resolved_checkpoint,
            inference_config = self._inference_config_path(module),
            categorical_features_indices = self.cat_indices or None,
            seed = self.args.seed,
        )
        for k in ('mix_precision', 'softmax_temperature', 'outlier_remove_std',
                  'average_before_softmax', 'test_batch_size', 'deterministic'):
            if k in general:
                kwargs[k] = general[k]
        self.model = module.LimiXPredictor(**kwargs)

    def _predict_raw(self, x_query):
        # Since v2.0 the task type is passed to predict() rather than the constructor.
        return self.model.predict(
            self.x_support, self.y_support, x_query,
            task_type="Regression" if self.is_regression else "Classification",
        )


class LimiXV21016Method(LimiXV2Method):
    """LimiX-2 with the publisher's retrained 1016 checkpoint."""

    HF_REPO = "stable-ai/LimiX-2-1016"
    CHECKPOINT = "LimiX-2-1016.ckpt"
    USE_ORIGINAL_CONFIG = False
