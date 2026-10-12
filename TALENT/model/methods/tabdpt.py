"""TabDPT method.

TabDPT (`Layer 6 AI <https://github.com/layer6ai-labs/TabDPT-inference>`_)
is a tabular foundation model that combines in-context learning with
retrieval and self-supervised pre-training on real data. Unlike TabPFN /
TabICL it does not have a hard context-size limit -- the retrieval step
selects the most relevant rows from the training set for each query.

The wrapper uses the upstream ``tabdpt`` package (sklearn-compatible
API: ``TabDPTClassifier`` / ``TabDPTRegressor``; weights are downloaded
automatically from the Hugging Face Hub) and exposes the runtime parameters
(``n_estimators`` -- TabDPT's ``n_ensembles`` --, ``context_size``,
``temperature``, ...) and the constructor options (``context_reduction``,
``normalizer``, ...) through ``config['general']`` so they can be tuned without
code changes. Each option is only forwarded to the upstream call that accepts
it, which keeps the wrapper working across the ``tabdpt`` 1.1 -> 1.2+ API
change.

Setup:
    pip install -U tabdpt
"""

import inspect

from TALENT.model.methods.foundation_base import SklearnFoundationMethod, resolve_n_estimators


class TabDPTMethod(SklearnFoundationMethod):
    """TabDPT (tabular foundation model with retrieval + ICL)."""

    def construct_model(self, model_config=None, cat_indices=None):
        try:
            from tabdpt import TabDPTClassifier, TabDPTRegressor
        except ImportError as e:
            raise ImportError(
                "TabDPT requires the `tabdpt` package. "
                "Install it via: pip install -U tabdpt"
            ) from e

        # TabDPT's constructors take few arguments at __init__ time --
        # most runtime knobs (n_ensembles, context_size, temperature, ...)
        # are passed to predict(). Persist them on self for later.
        general = self.args.config.get('general', {}) or {}
        self._predict_kwargs = {
            "n_ensembles": resolve_n_estimators(
                self.args, 8 if not self.is_regression else 2, "n_ensembles"
            ),
            "context_size": general.get("context_size", 2048),
            "seed": general.get("seed", self.args.seed),
        }
        if not self.is_regression:
            # Classification-only knobs
            self._predict_kwargs.setdefault("temperature", general.get("temperature", 0.8))
            self._predict_kwargs.setdefault(
                "permute_classes", general.get("permute_classes", True)
            )

        # Constructor options (tabdpt>=1.2). `context_reduction="retrieval"`
        # keeps the kNN-retrieved context of tabdpt<=1.1, whose default it was.
        init_kwargs = {"device": str(self.args.device)}
        for k in ("context_reduction", "normalizer", "missing_indicators", "clip_sigma",
                  "feature_reduction", "faiss_metric", "use_flash", "compile", "verbose"):
            if k in general:
                init_kwargs[k] = general[k]

        model_cls = TabDPTRegressor if self.is_regression else TabDPTClassifier
        self.model = model_cls(**self._supported_kwargs(model_cls.__init__, init_kwargs))

    @staticmethod
    def _supported_kwargs(fn, kwargs):
        """Keep only the kwargs that ``fn`` accepts.

        The upstream API changed between releases (e.g. ``n_ensembles`` moved
        from ``predict_proba`` to ``ensemble_predict_proba`` in tabdpt 1.2), so
        options are filtered per call instead of being dropped wholesale when
        one of them is unknown.
        """
        params = inspect.signature(fn).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return dict(kwargs)
        return {k: v for k, v in kwargs.items() if k in params}

    def _predict_raw(self, X):
        """Call the upstream predictor with the runtime knobs cached at
        construct time, forwarding each knob only if that call accepts it.

        Classification uses ``ensemble_predict_proba`` (tabdpt>=1.2), where
        ``n_ensembles`` / ``permute_classes`` live; older releases accepted
        them in ``predict_proba`` itself.
        """
        if self.is_regression:
            fn = self.model.predict
        else:
            fn = getattr(self.model, "ensemble_predict_proba", None) or self.model.predict_proba
        return fn(X, **self._supported_kwargs(fn, self._predict_kwargs))
