====================================
Core Components
====================================

Python runner
-------------

Use ``TALENT.run_from_dataset`` to load a named dataset from disk or
``TALENT.run`` to supply the train/validation and test splits directly.
Both return a ``RunResult``. Its ``predictions`` field retains each
method's native output. For probabilistic classifiers, use ``predict_proba``
for an ``(n_rows, n_classes)`` array and ``predict_labels`` for encoded class
indices. For regression, use ``predictions`` in the original target units.
See :doc:`../tutorials` for a runnable example.

Reproducible benchmarks
-----------------------

Use ``TALENT.benchmark.make_split`` to create a random, grouped or temporal
partition from stable row IDs. ``BenchmarkSplit.save`` and
``BenchmarkSplit.load`` persist the ordered IDs. ``with_context(k)`` varies
the training size while holding validation and test rows fixed. Call
``split.run(..., measure=True)`` to attach the split fingerprint and collect
per-seed measurements in ``result.measurements``. The existing runner also
accepts ``measure=True`` for datasets with their own splits.

``result_record`` extracts one metric from a run. ``compare_runs`` takes
these records and the planned dataset and split IDs. It returns coverage,
matched rankings, fit and prediction costs, and paired differences against a
reference method. A confidence interval requires at least two matched datasets.

Cross-validation
----------------

``TALENT.cross_validate(method, row_ids, N, C, y, info, cv=5)`` runs one
standard evaluation per outer fold. Classification defaults to stratified
folds. Supply ``groups`` for whole-group folds or ``timestamps`` for
expanding windows with an optional gap measured in distinct timestamps.
``threshold_cv`` optionally selects binary thresholds from inner out-of-fold
predictions. Each fit uses a separate validation partition for model selection.
The result exposes the outer and inner manifests, per-fold run results, total
wall time per fold, and row IDs aligned with out-of-fold predictions.
``TALENT.cross_validate_from_dataset`` evaluates the original training and
validation rows of a TALENT-format dataset and reserves its test split.

Pass ``tune=True, hpo_cv=3`` to score each Optuna candidate on three
held-out inner folds. The result's ``hpo_splits`` maps the enclosing split ID
to its scoring manifests. Set ``resume_dir`` to checkpoint completed fold
results and save each fit's Optuna study in SQLite. Rerunning the same
protocol there reuses completed work. ``resume_key`` can identify external
model revisions.

Every run reports ``fit_time``, ``hpo_time`` and ``predict_time`` in seconds.
Fit and prediction times are means across seeds. HPO time counts the shared
Optuna search once and is zero when tuning is disabled or a saved configuration
is reused. A CV result's ``hpo_time`` totals its outer and threshold-fold searches,
including recoverable trial time from interrupted studies. Set ``measure=True``
to add memory, tuning-call overhead and backend information.

Method infrastructure
---------------------

The core components are organized into three main categories:

**Essential Infrastructure:**

- **Utils**: Essential utilities for training, evaluation, configuration management, and system operations
- **Data**: Comprehensive data loading, preprocessing, transformation, and handling capabilities  
- **Method Base**: The foundational base class that all model implementations inherit from, providing unified interfaces

**Key Features:**

* **Unified Interface**: All components follow consistent APIs and patterns
* **Extensibility**: Easy to extend and customize for specific use cases
* **Robustness**: Comprehensive error handling and validation
* **Performance**: Optimized for efficiency in tabular data processing
* **Reproducibility**: Built-in support for deterministic operations

**Component Interactions:**

The core components are designed to work seamlessly together:

1. **Data Component** handles all data-related operations (loading, preprocessing, validation)
2. **Utils Component** provides supporting utilities (metrics, configuration, device management)
3. **Method Base** orchestrates the entire training/evaluation pipeline using Data and Utils

**Design Principles:**

* **Separation of Concerns**: Each component has well-defined responsibilities
* **Composition over Inheritance**: Components are composed rather than deeply inherited
* **Configuration-Driven**: Behavior is controlled through configuration rather than code changes
* **Type Safety**: Comprehensive type annotations and validation
* **Performance Optimization**: Efficient memory usage and computational patterns

Core Component Categories
-------------------------

Utility Functions and Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The utils module provides essential infrastructure for:

* **Training Infrastructure**: Optimizers, schedulers, and training loops
* **Evaluation Metrics**: Comprehensive metric computation for all task types
* **Configuration Management**: Loading, validation, and management of configurations
* **System Utilities**: Device management, path operations, and environment setup
* **Reproducibility**: Seed management and deterministic operations

Data Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

The data module implements a complete data processing pipeline:

* **Data Loading**: Support for multiple formats and sources
* **Preprocessing**: NaN handling, encoding, normalization, and feature engineering
* **Validation**: Data integrity checks and format validation
* **Transformation**: Feature transformations and augmentations
* **DataLoader Creation**: Efficient batch loading for training and inference

Method Infrastructure
~~~~~~~~~~~~~~~~~~~~~

The method base provides the foundational infrastructure:

* **Abstract Base Classes**: Common interfaces for all model implementations
* **Training Orchestration**: Complete training loop management
* **Evaluation Framework**: Standardized evaluation and metric computation
* **Checkpoint Management**: Model saving, loading, and resuming
* **Configuration Integration**: Seamless integration with configuration system

.. toctree::
   :maxdepth: 2

   utils
   data
   method_base
