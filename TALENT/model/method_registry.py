"""Unified method registry for TALENT.

Single source of truth for every model's metadata: architecture, hardware,
required preprocessing policy, output type, HPO support, etc. Replaces the
giant if/elif chain in `utils.get_method()` and the hand-written argparse
`choices` lists.

The registry is also the basis of the public Python API in `TALENT.api`,
where downstream code can introspect properties of a method without having
to read its source.

Each entry is derived from the actual `__init__` asserts in the method
file (e.g. `assert(args.cat_policy == 'indices')`) plus inspection of the
`predict()` return shape, so this is not aspirational — it documents what
TALENT already requires today.

Schema:
    name            : str  -- canonical model_type string used by the CLI
    module          : str  -- importable module path (lazy-loaded on use)
    class_name      : str  -- class name inside that module
    architecture    : Architecture.DEEP | Architecture.CLASSICAL
    hardware        : Hardware.GPU | Hardware.CPU
    output_type     : OutputType.LOGITS | PROBABILITIES | CLASS_LABELS
    cat_policy      : tuple[str, ...] of allowed cat_policy values, or
                      None for "no constraint" (default standard)
    normalization   : str | None -- forced normalization or None
    num_policy      : str | None -- forced num_policy or None
    supports_hpo    : bool
    supports_regression / supports_classification : bool
    train_row_limit : int | None -- soft cap on training-set size (e.g.
                      foundation models with limited context). The runner
                      and HPO loops respect this when sampling.
    notes           : str
"""

from __future__ import annotations

import importlib
import importlib.resources as _pkg_resources
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------------------------------------------------------
#  Bundled-resource resolution
# ----------------------------------------------------------------------------

def resolve_bundled_path(relative: str) -> Optional[str]:
    """Resolve a path relative to the installed ``TALENT`` package.

    Returns the absolute filesystem path if the resource exists, or
    ``None`` otherwise. Use this for bundled checkpoints and configs so
    that the package keeps working regardless of the user's current
    working directory (the historical hardcoded ``"./TALENT/..."`` paths
    only worked when run from the repository root).

    Example::

        path = resolve_bundled_path("model/models/models_tabpfn/tabpfn-v2-classifier.ckpt")
        # -> "/site-packages/TALENT/model/models/models_tabpfn/tabpfn-v2-classifier.ckpt"
        #    or None if not bundled (auto-download fallback)
    """
    try:
        import TALENT
        p = _pkg_resources.files(TALENT).joinpath(relative)
        # `Traversable` may be a filesystem path or a zipfile entry; convert
        # to str and check existence on the filesystem (not in a zip).
        path_str = str(p)
        if os.path.exists(path_str):
            return path_str
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------------
#  Enums
# ----------------------------------------------------------------------------

class Architecture(str, Enum):
    DEEP = "deep"
    CLASSICAL = "classical"


class Hardware(str, Enum):
    GPU = "gpu"
    CPU = "cpu"


class OutputType(str, Enum):
    """What `predict()` returns for classification tasks.

    Verified by reading each method's `predict()`:
      - LOGITS:        raw network output, needs softmax/sigmoid externally
      - PROBABILITIES: already in [0,1] summing to 1 (e.g. predict_proba)
      - CLASS_LABELS:  hard predictions (regression-only methods, or models
                       without probabilistic outputs)
    """
    LOGITS = "logits"
    PROBABILITIES = "probabilities"
    CLASS_LABELS = "class_labels"


# Sentinel: the special cat_policy '!indices' in asserts means "anything
# except indices". We encode this as the tuple of all *other* valid policies.
_ALL_CAT_POLICIES = ("ordinal", "ohe", "binary", "hash", "loo", "target",
                     "catboost", "tabr_ohe")  # everything except 'indices'


# ----------------------------------------------------------------------------
#  MethodSpec dataclass
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class MethodSpec:
    name: str
    module: str
    class_name: str
    architecture: Architecture
    hardware: Hardware
    output_type: OutputType
    cat_policy: Optional[Tuple[str, ...]] = None
    normalization: Optional[str] = None
    num_policy: Optional[str] = None
    supports_hpo: bool = True
    supports_regression: bool = True
    supports_classification: bool = True
    train_row_limit: Optional[int] = None
    notes: str = ""

    def get_class(self):
        """Lazy-import the method class from its module."""
        mod = importlib.import_module(self.module)
        return getattr(mod, self.class_name)

    def validate_args(self, args) -> None:
        """Raise ValueError if `args` violates this spec's preprocessing
        constraints. Used by the high-level runner to fail fast with a
        readable message instead of an obscure `assert`.
        """
        problems: List[str] = []
        if self.cat_policy is not None and getattr(args, 'cat_policy', None) not in self.cat_policy:
            problems.append(
                f"cat_policy must be one of {self.cat_policy}, got "
                f"{getattr(args, 'cat_policy', None)!r}"
            )
        if self.normalization is not None and getattr(args, 'normalization', None) != self.normalization:
            problems.append(
                f"normalization must be {self.normalization!r}, got "
                f"{getattr(args, 'normalization', None)!r}"
            )
        if self.num_policy is not None and getattr(args, 'num_policy', None) != self.num_policy:
            problems.append(
                f"num_policy must be {self.num_policy!r}, got "
                f"{getattr(args, 'num_policy', None)!r}"
            )
        if not self.supports_hpo and getattr(args, 'tune', False):
            problems.append(f"{self.name} does not support HPO (set tune=False)")
        if problems:
            raise ValueError(
                f"Invalid args for method {self.name!r}:\n  - " + "\n  - ".join(problems)
            )


# ----------------------------------------------------------------------------
#  Registry data
# ----------------------------------------------------------------------------

def _spec(name, module, class_name, architecture, hardware, output_type, **kw):
    return MethodSpec(
        name=name, module=module, class_name=class_name,
        architecture=architecture, hardware=hardware, output_type=output_type,
        **kw,
    )


# Convenience shortcuts to keep entries readable
_DEEP = Architecture.DEEP
_CLASSICAL = Architecture.CLASSICAL
_GPU = Hardware.GPU
_CPU = Hardware.CPU
_LOGITS = OutputType.LOGITS
_PROBS = OutputType.PROBABILITIES
_LABELS = OutputType.CLASS_LABELS


_METHODS_DEEP = [
    # ----- Basic neural -----
    _spec("mlp", "TALENT.model.methods.mlp", "MLPMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=_ALL_CAT_POLICIES),
    _spec("resnet", "TALENT.model.methods.resnet", "ResNetMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=_ALL_CAT_POLICIES),
    _spec("snn", "TALENT.model.methods.snn", "SNNMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=("indices",)),
    _spec("realmlp", "TALENT.model.methods.realmlp", "RealMLPMethod",
          _DEEP, _GPU, _PROBS, cat_policy=("indices",),
          notes="Classifier predict() returns predict_proba output."),
    _spec("mlp_plr", "TALENT.model.methods.mlp_plr", "MLP_PLRMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=("tabr_ohe",), num_policy=None),

    # ----- Transformer-based -----
    _spec("autoint", "TALENT.model.methods.autoint", "AutoIntMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=("indices",)),
    _spec("saint", "TALENT.model.methods.saint", "SaintMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=("indices",)),
    _spec("ftt", "TALENT.model.methods.ftt", "FTTMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=("indices",)),
    _spec("tabtransformer", "TALENT.model.methods.tabtransformer", "TabTransformerMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=("indices",)),
    _spec("excelformer", "TALENT.model.methods.excelformer", "ExcelFormerMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=_ALL_CAT_POLICIES),
    _spec("t2gformer", "TALENT.model.methods.t2gformer", "T2GFormerMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=("indices",)),
    _spec("amformer", "TALENT.model.methods.amformer", "AMFormerMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=("indices",)),
    _spec("trompt", "TALENT.model.methods.trompt", "TromptMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=("indices",)),

    # ----- Tree-mimic / specialized -----
    _spec("dcn2", "TALENT.model.methods.dcn2", "DCN2Method",
          _DEEP, _GPU, _LOGITS, cat_policy=("indices",)),
    _spec("node", "TALENT.model.methods.node", "NodeMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=_ALL_CAT_POLICIES),
    _spec("tabcaps", "TALENT.model.methods.tabcaps", "TabCapsMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=_ALL_CAT_POLICIES,
          supports_regression=False),
    _spec("tabnet", "TALENT.model.methods.tabnet", "TabNetMethod",
          _DEEP, _GPU, _PROBS, cat_policy=_ALL_CAT_POLICIES,
          notes="Classifier predict() returns predict_proba output."),
    _spec("danets", "TALENT.model.methods.danets", "DANetsMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=_ALL_CAT_POLICIES),
    _spec("grownet", "TALENT.model.methods.grownet", "GrowNetMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=("indices",)),
    _spec("grande", "TALENT.model.methods.grande", "GRANDEMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=("indices",)),
    _spec("tabm", "TALENT.model.methods.tabm", "TabMMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=("indices",)),

    # ----- KNN-style -----
    _spec("tabr", "TALENT.model.methods.tabr", "TabRMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=("tabr_ohe",), num_policy="none"),
    _spec("modernNCA", "TALENT.model.methods.modernNCA", "ModernNCAMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=("tabr_ohe",), num_policy="none"),
    _spec("dnnr", "TALENT.model.methods.dnnr", "DNNRMethod",
          _DEEP, _GPU, _LABELS, cat_policy=_ALL_CAT_POLICIES,
          supports_classification=False,
          notes="Regression-only KNN-based method."),

    # ----- Regularization-based -----
    _spec("tangos", "TALENT.model.methods.tangos", "TangosMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=_ALL_CAT_POLICIES),
    _spec("switchtab", "TALENT.model.methods.switchtab", "SwitchTabMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=_ALL_CAT_POLICIES),
    _spec("ptarl", "TALENT.model.methods.ptarl", "PTARLMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=("indices",)),
    _spec("bishop", "TALENT.model.methods.bishop", "BiSHopMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=("indices",)),
    _spec("protogate", "TALENT.model.methods.protogate", "ProtoGateMethod",
          _DEEP, _GPU, _PROBS, cat_policy=_ALL_CAT_POLICIES,
          supports_regression=False,
          notes="ProtoGate predict() returns neighbor-vote probabilities."),
    _spec("tabautopnpnet", "TALENT.model.methods.tabautopnpnet", "TabAutoPNPNetMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=("tabr_ohe",)),

    # ----- Foundation models -----
    _spec("tabpfn", "TALENT.model.methods.tabpfn", "TabPFNMethod",
          _DEEP, _GPU, _PROBS, cat_policy=("indices",),
          normalization="none", num_policy="none",
          supports_hpo=False, supports_regression=False,
          train_row_limit=1_000,
          notes="TabPFN v1 (bundled). Up to 1k samples; classification only."),
    _spec("tabpfn_v2", "TALENT.model.methods.tabpfn_v2", "TabPFNMethod",
          _DEEP, _GPU, _PROBS, cat_policy=("indices",),
          normalization="none", num_policy="none",
          supports_hpo=False, train_row_limit=10_000,
          notes="TabPFN v2 (bundled, Nature 2025)."),
    _spec("tabpfn_v2_5", "TALENT.model.methods.tabpfn_v2_5", "TabPFNv2_5Method",
          _DEEP, _GPU, _PROBS, cat_policy=("indices",),
          normalization="none", num_policy="none",
          supports_hpo=False, train_row_limit=50_000,
          notes="TabPFN v2.5 (PriorLabs Nov 2025; external `tabpfn>=8.0.0`). "
                "~50k rows x 2k features native context."),
    _spec("tabpfn_v3", "TALENT.model.methods.tabpfn_v3", "TabPFNv3Method",
          _DEEP, _GPU, _PROBS, cat_policy=("indices",),
          normalization="none", num_policy="none",
          supports_hpo=False, train_row_limit=1_000_000,
          notes="TabPFN v3 (external `tabpfn>=8.0.0`). ~1M-row context."),
    _spec("tabpfn_real", "TALENT.model.methods.tabpfn_real", "TabPFNRealMethod",
          _DEEP, _GPU, _PROBS, cat_policy=("indices",),
          normalization="none", num_policy="none",
          supports_hpo=False, train_row_limit=10_000,
          notes="Real-TabPFN (continued pre-training on real datasets)."),
    _spec("hyperfast", "TALENT.model.methods.hyperfast", "HyperFastMethod",
          _DEEP, _GPU, _PROBS, cat_policy=("indices",),
          normalization="none", num_policy="none",
          supports_hpo=False, supports_regression=False,
          notes="HyperFast meta-trained hypernetwork; classification only."),
    _spec("tabptm", "TALENT.model.methods.tabptm", "TabPTMMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=("ohe",),
          normalization="standard", num_policy="none",
          supports_hpo=False),
    _spec("tabicl", "TALENT.model.methods.tabicl", "TabICLMethod",
          _DEEP, _GPU, _PROBS, cat_policy=("indices",),
          normalization="none", num_policy="none",
          supports_hpo=False, supports_regression=False,
          train_row_limit=500_000,
          notes="TabICL v1.1 (bundled); classification only — use tabicl_v2 for regression."),
    _spec("tabicl_v2", "TALENT.model.methods.tabicl_v2", "TabICLv2Method",
          _DEEP, _GPU, _PROBS, cat_policy=("indices",),
          normalization="none", num_policy="none",
          supports_hpo=False, train_row_limit=1_000_000,
          notes="TabICL v2 (external `tabicl>=2.0.0`); supports regression."),
    _spec("mitra", "TALENT.model.methods.mitra", "MitraMethod",
          _DEEP, _GPU, _LOGITS, cat_policy=("indices",),
          normalization="none", num_policy="none",
          supports_hpo=False, train_row_limit=10_000,
          notes="Mitra (Amazon Science). ICL with cross-attention; O(N_train * N_test) memory."),
    _spec("limix", "TALENT.model.methods.limix", "LimiXMethod",
          _DEEP, _GPU, _PROBS, cat_policy=("indices",),
          normalization="none", num_policy="none",
          supports_hpo=False,
          notes="LimiX tabular foundation model; classifier returns probabilities."),
    _spec("tabdpt", "TALENT.model.methods.tabdpt", "TabDPTMethod",
          _DEEP, _GPU, _PROBS, cat_policy=("indices",),
          normalization="none", num_policy="none",
          supports_hpo=False,
          notes="TabDPT (Layer 6 AI). ICL + retrieval; supports both "
                "classification and regression. External `tabdpt` package."),
    _spec("tabfm", "TALENT.model.methods.tabfm", "TabFMMethod",
          _DEEP, _GPU, _PROBS, cat_policy=("indices",),
          normalization="none", num_policy="none",
          supports_hpo=False,
          notes="TabFM v1.0.0 (Google Research). Zero-shot ICL foundation "
                "model via optional `tabfm[pytorch]`; classification is "
                "limited to 10 classes by the upstream model."),
]


_METHODS_CLASSICAL = [
    _spec("dummy", "TALENT.model.classical_methods.dummy", "DummyMethod",
          _CLASSICAL, _CPU, _PROBS, cat_policy=_ALL_CAT_POLICIES,
          notes="Baseline: majority-class / mean predictor."),
    _spec("LogReg", "TALENT.model.classical_methods.logreg", "LogRegMethod",
          _CLASSICAL, _CPU, _PROBS, cat_policy=_ALL_CAT_POLICIES,
          supports_regression=False),
    _spec("LinearRegression", "TALENT.model.classical_methods.lr", "LinearRegressionMethod",
          _CLASSICAL, _CPU, _LABELS, cat_policy=_ALL_CAT_POLICIES,
          supports_classification=False, supports_hpo=False,
          notes="Regression-only; returns point estimates."),
    _spec("xgboost", "TALENT.model.classical_methods.xgboost", "XGBoostMethod",
          _CLASSICAL, _CPU, _PROBS, cat_policy=_ALL_CAT_POLICIES,
          notes="GPU-capable when CUDA available (see utils.tune_hyper_parameters)."),
    _spec("catboost", "TALENT.model.classical_methods.catboost", "CatBoostMethod",
          _CLASSICAL, _CPU, _PROBS, cat_policy=("indices",),
          notes="Uses native categorical handling via cat_policy='indices'."),
    _spec("lightgbm", "TALENT.model.classical_methods.lightgbm", "LightGBMMethod",
          _CLASSICAL, _CPU, _PROBS, cat_policy=_ALL_CAT_POLICIES),
    _spec("RandomForest", "TALENT.model.classical_methods.randomforest", "RandomForestMethod",
          _CLASSICAL, _CPU, _PROBS, cat_policy=_ALL_CAT_POLICIES),
    _spec("svm", "TALENT.model.classical_methods.svm", "SvmMethod",
          _CLASSICAL, _CPU, _PROBS, cat_policy=_ALL_CAT_POLICIES,
          notes="LinearSVC + CalibratedClassifierCV for probability calibration."),
    _spec("knn", "TALENT.model.classical_methods.knn", "KnnMethod",
          _CLASSICAL, _CPU, _PROBS, cat_policy=_ALL_CAT_POLICIES),
    _spec("NCM", "TALENT.model.classical_methods.ncm", "NCMMethod",
          _CLASSICAL, _CPU, _PROBS, cat_policy=_ALL_CAT_POLICIES,
          supports_regression=False,
          notes="Nearest class mean with softmax over negative distances."),
    _spec("NaiveBayes", "TALENT.model.classical_methods.naivebayes", "NaiveBayesMethod",
          _CLASSICAL, _CPU, _PROBS, cat_policy=_ALL_CAT_POLICIES,
          supports_regression=False),
    _spec("rfm", "TALENT.model.classical_methods.rfm", "RFMMethod",
          _CLASSICAL, _GPU, _PROBS, cat_policy=("ohe",), normalization="standard",
          notes="RFM (Recursive Feature Machines). GPU-accelerated kernel method."),
    _spec("xrfm", "TALENT.model.classical_methods.xrfm", "XRFMMethod",
          _CLASSICAL, _GPU, _PROBS, cat_policy=("ohe",), normalization="standard",
          notes="xRFM (RFMs + adaptive tree structure)."),
]


# Master registry: name -> MethodSpec
METHOD_REGISTRY: Dict[str, MethodSpec] = {
    spec.name: spec for spec in (_METHODS_DEEP + _METHODS_CLASSICAL)
}


# ----------------------------------------------------------------------------
#  Public lookup helpers
# ----------------------------------------------------------------------------

def get_method_spec(name: str) -> MethodSpec:
    """Return the MethodSpec for `name`, or raise KeyError with a helpful list."""
    if name not in METHOD_REGISTRY:
        # Be lenient about case for a couple of historical name variants.
        for canonical in METHOD_REGISTRY:
            if canonical.lower() == name.lower():
                return METHOD_REGISTRY[canonical]
        raise KeyError(
            f"Unknown method {name!r}. Available: "
            f"{sorted(METHOD_REGISTRY.keys())}"
        )
    return METHOD_REGISTRY[name]


def get_method_class(name: str):
    """Return the method class (lazy import). Replaces the if/elif chain."""
    return get_method_spec(name).get_class()


def list_methods(
    *,
    architecture: Optional[Architecture] = None,
    hardware: Optional[Hardware] = None,
    supports_regression: Optional[bool] = None,
    supports_classification: Optional[bool] = None,
    supports_hpo: Optional[bool] = None,
) -> List[MethodSpec]:
    """List registered methods, optionally filtered by capability.

    Each filter is applied only if the kwarg is non-None.
    """
    out: List[MethodSpec] = []
    for spec in METHOD_REGISTRY.values():
        if architecture is not None and spec.architecture != architecture:
            continue
        if hardware is not None and spec.hardware != hardware:
            continue
        if supports_regression is not None and spec.supports_regression != supports_regression:
            continue
        if supports_classification is not None and spec.supports_classification != supports_classification:
            continue
        if supports_hpo is not None and spec.supports_hpo != supports_hpo:
            continue
        out.append(spec)
    return out


def deep_method_names() -> List[str]:
    """All deep-method names, in registration order. Used for argparse choices."""
    return [s.name for s in _METHODS_DEEP]


def classical_method_names() -> List[str]:
    """All classical-method names, in registration order. Used for argparse choices."""
    return [s.name for s in _METHODS_CLASSICAL]


def all_method_names() -> List[str]:
    return deep_method_names() + classical_method_names()
