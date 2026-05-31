"""TALENT — A Tabular Analytics and Learning Toolbox.

Two ways to use TALENT:

1. **CLI scripts** (original, unchanged):

   .. code-block:: bash

       python train_model_deep.py --model_type tabpfn_v3
       python train_model_classical.py --model_type catboost

2. **Python API** (added 2026, library mode):

   .. code-block:: python

       import TALENT
       from TALENT.model.lib.data import get_dataset

       train_val, test, info = get_dataset("housing", "./data")
       result = TALENT.run("tabpfn_v3", train_val, test, info)
       print(result.metrics, result.metric_names)

       # Introspect:
       spec = TALENT.get_method_spec("tabpfn_v3")
       print(spec.cat_policy, spec.supports_hpo, spec.train_row_limit)

       # Enumerate available methods:
       for s in TALENT.list_methods(architecture=TALENT.Architecture.DEEP):
           print(s.name)

Both surfaces share a single :class:`MethodSpec` registry, so adding a new
method requires only a registry entry (no changes to CLI or API code).
"""

__version__ = "0.1.0"

# Lazy import surface: importing TALENT does NOT pull in torch.
# Each name is resolved on first access via `__getattr__`.

_LAZY_API = {
    "run": "TALENT.api",
    "run_from_dataset": "TALENT.api",
    "build_args": "TALENT.api",
    "RunResult": "TALENT.api",
}

_LAZY_REGISTRY = {
    "MethodSpec": "TALENT.model.method_registry",
    "METHOD_REGISTRY": "TALENT.model.method_registry",
    "Architecture": "TALENT.model.method_registry",
    "Hardware": "TALENT.model.method_registry",
    "OutputType": "TALENT.model.method_registry",
    "get_method_spec": "TALENT.model.method_registry",
    "get_method_class": "TALENT.model.method_registry",
    "list_methods": "TALENT.model.method_registry",
    "deep_method_names": "TALENT.model.method_registry",
    "classical_method_names": "TALENT.model.method_registry",
    "all_method_names": "TALENT.model.method_registry",
}


def __getattr__(name):
    """Lazy attribute resolution. Loads the right submodule on first access."""
    import importlib
    target = _LAZY_REGISTRY.get(name) or _LAZY_API.get(name)
    if target is None:
        raise AttributeError(f"module 'TALENT' has no attribute {name!r}")
    mod = importlib.import_module(target)
    obj = getattr(mod, name)
    globals()[name] = obj  # cache so subsequent attribute access is fast
    return obj


__all__ = ["__version__"] + sorted(set(_LAZY_REGISTRY) | set(_LAZY_API))
