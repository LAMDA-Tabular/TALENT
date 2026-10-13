import inspect

from TALENT.model.methods.foundation_base import SklearnFoundationMethod


class TabICLv2Method(SklearnFoundationMethod):
    """TabICL v2 method — supports both classification AND regression.

    Uses the official `tabicl>=2.0.0` package which adds regression support
    via `TabICLRegressor` (v1 was classification-only). The checkpoints are
    downloaded automatically from the Hugging Face Hub (``jingang/TabICL``).

    Setup:
        pip install -U 'tabicl>=2.0.0'
    """

    def construct_model(self, model_config=None, cat_indices=None):
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
        # (class_shuffle_method / support_many_classes are the tabicl>=2.1 names of
        # class_shift / use_hierarchical; only the names the installed version
        # accepts are forwarded.)
        for k in ('checkpoint_version', 'model_path', 'norm_methods',
                  'feat_shuffle_method', 'outlier_threshold', 'inference_config',
                  'kv_cache', 'offload_mode', 'class_shuffle_method', 'support_many_classes'):
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
