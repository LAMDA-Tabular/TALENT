"""TabPFN-3.5 method.

TabPFN-3.5 (PriorLabs, September 2026) is the successor of TabPFN-3, with
native context up to ~1M rows and up to 20 000 features. A single checkpoint
serves both classification and regression. The alpha "Fast" checkpoint
(TabPFN-3.5-Fast, up to ~6x faster inference) is selected with
``config['general']['fast'] = true``.

The open weights are released under a non-commercial license that must be
accepted once. For headless runs (e.g. a cluster), accept the license on the
Prior Labs platform and set ``TABPFN_TOKEN=<api key>`` (and optionally
``TABPFN_NO_BROWSER=1``) before the first download.

Setup:
    pip install -U 'tabpfn>=9.0.0'
"""

from TALENT.model.methods.tabpfn_v3 import TabPFNv3Method


class TabPFNv3_5Method(TabPFNv3Method):
    """TabPFN-3.5 (in-context tabular foundation model).

    Uses the official `tabpfn>=9.0.0` package with the model version pinned
    via ``ModelVersion.V3_5`` (or ``V3_5_FAST``).
    """

    MODEL_VERSION = "V3_5"
    MODEL_NAME = "TabPFN-3.5"
    PACKAGE_REQUIREMENT = "tabpfn>=9.0.0"
    # Older packages do not know TabPFN-3.5, so never fall back to their default.
    ALLOW_UNPINNED_FALLBACK = False

    def construct_model(self, model_config=None, cat_indices=None):
        general = self.args.config.get('general', {}) or {}
        self.MODEL_VERSION = "V3_5_FAST" if general.get('fast', False) else "V3_5"
        super().construct_model(model_config, cat_indices)
