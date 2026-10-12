from TALENT.model.methods.foundation_base import SklearnFoundationMethod, resolve_n_estimators


class TabICLMethod(SklearnFoundationMethod):
    """TabICL v1.1 (bundled copy of the TabICL library; classification only).

    The checkpoint is used from ``TALENT/model/models/models_tabicl/`` when
    bundled, and otherwise downloaded automatically from the Hugging Face Hub
    (``jingang/TabICL-clf``). Use ``tabicl_v2`` for regression.
    """

    def __init__(self, args, is_regression):
        # The bundled TabICL v1.1 is classifier-only — use `tabicl_v2` for regression.
        assert is_regression is False, (
            "TabICL v1 does not support regression. "
            "Use --model_type tabicl_v2 (requires `pip install -U 'tabicl>=2.0.0'`)."
        )
        super().__init__(args, is_regression)

    def construct_model(self, model_config=None, cat_indices=None):
        from TALENT.model.lib.tabicl.classifier import TabICLClassifier
        from TALENT.model.method_registry import resolve_bundled_path
        # Use bundled checkpoint if present; otherwise the TabICL library
        # auto-downloads via `model_path=None` + `allow_auto_download=True`.
        model_path = resolve_bundled_path(
            "model/models/models_tabicl/tabicl-classifier-v1.1-0506.ckpt"
        )
        self.model = TabICLClassifier(
            device=self.args.device,
            random_state=self.args.seed,
            checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",
            model_path=model_path,
            n_estimators=resolve_n_estimators(self.args, 32),  # number of ensemble members
            norm_methods=["none", "power"],   # normalization methods to try
            feat_shuffle_method="latin",      # feature permutation strategy
            class_shift=True,                 # whether to apply cyclic shifts to class labels
            outlier_threshold=4.0,            # z-score threshold for outlier detection and clipping
            softmax_temperature=0.9,          # controls prediction confidence
            average_logits=True,              # whether ensemble averaging is done on logits or probabilities
            use_hierarchical=True,            # enable hierarchical classification for datasets with many classes
            batch_size=8,                     # process this many ensemble members together (reduce RAM usage)
            use_amp=True,                     # use automatic mixed precision for faster inference
            allow_auto_download=True,         # whether automatic download to the specified path is allowed
            n_jobs=None,                      # number of threads to use for PyTorch
            verbose=False,                    # print progress messages
            inference_config=None,            # inference configuration for fine-grained control
        )
