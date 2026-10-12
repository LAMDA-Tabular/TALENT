"""TabPFN v2.5 method.

TabPFN v2.5 (PriorLabs, November 2025) is the intermediate release between
v2 (Nature 2025) and v3 (2026). It scales in-context learning to ~50 000
rows × 2 000 features and is a drop-in replacement for v2 with better
accuracy and speed.

Like v3, v2.5 is distributed through the upstream `tabpfn>=8.0.0`
package. The wrapper shares its implementation with the v3 wrapper and only
pins a different ``ModelVersion`` (``V2_5``).

Setup:
    pip install -U 'tabpfn>=8.0.0'
"""

from TALENT.model.methods.tabpfn_v3 import TabPFNv3Method


class TabPFNv2_5Method(TabPFNv3Method):
    """TabPFN v2.5 (in-context tabular foundation model).

    Uses the official `tabpfn>=8.0.0` package with the model version pinned
    to v2.5 via ``ModelVersion.V2_5``.
    """

    MODEL_VERSION = "V2_5"
    MODEL_NAME = "TabPFN v2.5"
    PACKAGE_REQUIREMENT = "tabpfn>=8.0.0"
    # v2.5 has never been the package default, so the default constructor
    # would load a different model: fail loudly instead of falling back.
    ALLOW_UNPINNED_FALLBACK = False
