"""Optuna studies for cross-validated and restartable searches."""

from copy import deepcopy
from math import isfinite
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory

import optuna
from optuna.trial import TrialState

from .measurement import elapsed_call
from .scoring import Scorer


def run_trials(study, objective, args, **options):
    """Time the Optuna search once, including candidate fits and scoring."""
    _, args.hpo_time = elapsed_call(
        "hpo_seconds", getattr(args, "device", None), study.optimize, objective, **options
    )


def optimize_candidates(args, info, train_val, direction, sample_config, score_candidate, persist, recovery):
    """Keep the resolved configuration and the individual scores with each trial."""
    from TALENT.model.utils import get_method, set_seeds

    budget = args.n_trials
    if isinstance(budget, bool) or not isinstance(budget, Integral) or budget < 1:
        raise ValueError("n_trials must be a positive integer")
    storage = None
    if persist:
        db = Path(args.save_path).resolve() / "study.sqlite3"
        import sqlite3
        storage = optuna.storages.RDBStorage(
            url="sqlite://", engine_kwargs={"creator": lambda: sqlite3.connect(str(db), timeout=30)}
        )
    study = optuna.create_study(storage=storage, study_name="tuning", direction=direction,
                                sampler=optuna.samplers.TPESampler(seed=0), load_if_exists=persist)
    try:
        history = study.get_trials(deepcopy=False)
        if recovery is not None:
            recovery["trial_seconds"] = sum(
                t.duration.total_seconds() for t in history if t.duration is not None and
                (t.state == TrialState.COMPLETE or (t.state == TrialState.FAIL and "error" in t.user_attrs))
            )
        for old in history:
            if old.state == TrialState.RUNNING:
                # The benchmark directory is exclusively locked by the current worker.
                study.tell(old.number, state=TrialState.FAIL)
        used = sum(t.state == TrialState.COMPLETE or
                   (t.state == TrialState.FAIL and "error" in t.user_attrs) for t in history)
        if history:
            # A fresh sampler seeded at zero would replay its startup draws.
            study = optuna.load_study(storage=storage, study_name="tuning",
                                      sampler=optuna.samplers.TPESampler(seed=len(history)))

        def objective(trial):
            try:
                config = sample_config(trial)
                trial.set_user_attr("config", config)
                with TemporaryDirectory(prefix="candidate-", dir=args.save_path) as work:
                    trial_args = deepcopy(args)
                    trial_args.save_path = work
                    if score_candidate is not None:
                        return score_candidate(trial, trial_args, config)
                    set_seeds(trial_args.seed)
                    scorer = getattr(trial_args, "tune_metric", None)
                    if isinstance(scorer, Scorer):
                        trial_args.tune_metric = None
                    method = get_method(args.model_type)(trial_args, info["task_type"] == "regression")
                    method.fit(train_val, info, train=True, config=deepcopy(config))
                    if isinstance(scorer, Scorer):
                        from TALENT.model.lib.evaluation import evaluate, _val_as_test
                        from TALENT.model.method_registry import get_method_spec

                        result = evaluate(method, train_val, _val_as_test(train_val), info,
                                          trial_args.evaluate_option, tune_threshold=False, scoring=scorer,
                                          output_type=get_method_spec(args.model_type).output_type)
                        return result["metrics"][-1]
                    score = float(method.trlog["best_res"])
                    if not isfinite(score):
                        raise ValueError("non-finite validation objective")
                    return score
            except Exception as exc:
                trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
                raise

        if budget > used:
            run_trials(study, objective, args, n_trials=budget - used, catch=(Exception,))
        if not study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
            failures = study.get_trials(deepcopy=False, states=(TrialState.FAIL,))
            reason = failures[-1].user_attrs.get("error", "interrupted trial") if failures else "no trials"
            raise RuntimeError(f"No Optuna trial completed successfully: {reason}")
        return deepcopy(study.best_trial.user_attrs["config"])
    finally:
        # Release SQLite handles promptly (including on Windows).
        if storage is not None:
            storage.remove_session()
            storage.engine.dispose()
