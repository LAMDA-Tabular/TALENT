from TALENT.model.methods.foundation_base import SklearnFoundationMethod


class TabPFNv3Method(SklearnFoundationMethod):
    """TabPFN v3 method.

    Uses the official `tabpfn>=8.0.0` package. The model version is pinned via
    ``ModelVersion.V3`` because the package's *default* model moves forward
    (``tabpfn`` 9.0 switched it from TabPFN-3 to TabPFN-3.5), so an unpinned
    constructor would silently run a different model.

    Setup:
        pip install -U tabpfn>=8.0.0
    """

    # ``tabpfn.constants.ModelVersion`` member to pin, and how to describe it.
    MODEL_VERSION = "V3"
    MODEL_NAME = "TabPFN v3"
    PACKAGE_REQUIREMENT = "tabpfn>=8.0.0"
    # TabPFN-3 is the package default in tabpfn 8.x, so if the version cannot
    # be pinned (older 8.x without ``create_default_for_version``) the default
    # constructor still loads the right model.
    ALLOW_UNPINNED_FALLBACK = True

    def _build_kwargs(self, cat_indices):
        general = self.args.config.get('general', {}) or {}
        kwargs = dict(
            device=self.args.device,
            random_state=self.args.seed,
            ignore_pretraining_limits=general.get('ignore_pretraining_limits', True),
            categorical_features_indices=cat_indices,
        )
        # Optional knobs forwarded if present in config — keeps the wrapper tolerant
        # to minor signature changes between tabpfn releases.
        for k in ('n_estimators', 'softmax_temperature', 'memory_saving_mode',
                  'fit_mode', 'inference_precision', 'model_path'):
            if k in general:
                kwargs[k] = general[k]
        return kwargs

    def construct_model(self, model_config=None, cat_indices=None):
        cat_indices = cat_indices or []
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
        except ImportError as e:
            raise ImportError(
                f"{self.MODEL_NAME} requires the `{self.PACKAGE_REQUIREMENT}` package. "
                f"Install it via: pip install -U '{self.PACKAGE_REQUIREMENT}'."
            ) from e

        # Default ensemble sizes mirror upstream defaults; users can override via config['general'].
        general = self.args.config.get('general', {}) or {}
        n_est_default = 8 if self.is_regression else 4
        kwargs = self._build_kwargs(cat_indices)
        kwargs.setdefault('n_estimators', general.get('n_estimators', n_est_default))

        model_cls = TabPFNRegressor if self.is_regression else TabPFNClassifier
        try:
            from tabpfn.constants import ModelVersion
            version = getattr(ModelVersion, self.MODEL_VERSION, None)
        except ImportError:
            version = None
        if version is not None and hasattr(model_cls, 'create_default_for_version'):
            # kwargs are applied as overrides, so a configured `model_path`
            # or `n_estimators` still takes precedence.
            self.model = model_cls.create_default_for_version(version, **kwargs)
        elif self.ALLOW_UNPINNED_FALLBACK:
            self.model = model_cls(**kwargs)
        else:
            raise ImportError(
                f"{self.MODEL_NAME} is not available in the installed `tabpfn` package "
                f"(ModelVersion.{self.MODEL_VERSION} not found). "
                f"Install it via: pip install -U '{self.PACKAGE_REQUIREMENT}'."
            )
