**PFN v2**
==========

TabPFN v2 provides TabPFN classifier and regressor implementations for tabular data.


Functions
~~~~~~~~~

.. code-block:: python

    class TabPFNClassifier(ClassifierMixin, BaseEstimator)

TabPFN classifier for tabular data classification.

**Parameters:**

* **n_estimators** *(int, optional, Default is 4)* - Number of estimators for ensemble.
* **categorical_features_indices** *(Sequence[int] | None, optional, Default is None)* - Indices of categorical features.
* **softmax_temperature** *(float, optional, Default is 0.9)* - Temperature for softmax function.
* **balance_probabilities** *(bool, optional, Default is False)* - Whether to balance probabilities.
* **average_before_softmax** *(bool, optional, Default is False)* - Whether to average before softmax.
* **model_path** *(str | Path | Literal["auto"], optional, Default is "auto")* - Path to model checkpoint.
* **device** *(str | torch.device | Literal["auto"], optional, Default is "auto")* - Device for computation.
* **ignore_pretraining_limits** *(bool, optional, Default is False)* - Whether to ignore pretraining limits.
* **inference_precision** *(dtype | Literal["autocast", "auto"], optional, Default is "auto")* - Inference precision.
* **fit_mode** *(Literal["low_memory", "fit_preprocessors", "fit_with_cache"], optional, Default is "fit_preprocessors")* - Fit mode.
* **memory_saving_mode** *(bool | Literal["auto"] | float | int, optional, Default is "auto")* - Memory saving mode.
* **random_state** *(int | np.random.RandomState | np.random.Generator | None, optional, Default is 0)* - Random seed.
* **n_jobs** *(int, optional, Default is -1)* - Number of jobs for parallel processing.
* **inference_config** *(dict | ModelInterfaceConfig | None, optional, Default is None)* - Inference configuration.

**Methods:**

* **fit(self, X, y)** - Fit the classifier.
* **predict(self, X)** - Predict class labels.
* **predict_proba(self, X)** - Predict class probabilities.


.. code-block:: python

    class TabPFNRegressor(RegressorMixin, BaseEstimator)

TabPFN regressor for tabular data regression.

**Parameters:**

* **n_estimators** *(int, optional, Default is 4)* - Number of estimators for ensemble.
* **categorical_features_indices** *(Sequence[int] | None, optional, Default is None)* - Indices of categorical features.
* **model_path** *(str | Path | Literal["auto"], optional, Default is "auto")* - Path to model checkpoint.
* **device** *(str | torch.device | Literal["auto"], optional, Default is "auto")* - Device for computation.
* **ignore_pretraining_limits** *(bool, optional, Default is False)* - Whether to ignore pretraining limits.
* **inference_precision** *(dtype | Literal["autocast", "auto"], optional, Default is "auto")* - Inference precision.
* **fit_mode** *(Literal["low_memory", "fit_preprocessors", "fit_with_cache"], optional, Default is "fit_preprocessors")* - Fit mode.
* **memory_saving_mode** *(bool | Literal["auto"] | float | int, optional, Default is "auto")* - Memory saving mode.
* **random_state** *(int | np.random.RandomState | np.random.Generator | None, optional, Default is 0)* - Random seed.
* **n_jobs** *(int, optional, Default is -1)* - Number of jobs for parallel processing.
* **inference_config** *(dict | ModelInterfaceConfig | None, optional, Default is None)* - Inference configuration.

**Methods:**

* **fit(self, X, y)** - Fit the regressor.
* **predict(self, X)** - Predict target values.


.. code-block:: python

    class InferenceEngine

Inference engine for TabPFN models.

**Parameters:**

* **model** - TabPFN model.
* **config** - Inference configuration.
* **device** *(torch.device)* - Device for computation.

**Methods:**

* **predict(self, X)** - Make predictions.
* **predict_proba(self, X)** - Predict probabilities.


.. code-block:: python

    class EnsembleConfig

Configuration for ensemble models.

**Parameters:**

* **n_estimators** *(int)* - Number of estimators.
* **categorical_features_indices** *(List[int])* - Indices of categorical features.
* **softmax_temperature** *(float)* - Softmax temperature.
* **balance_probabilities** *(bool)* - Whether to balance probabilities.


.. code-block:: python

    class ClassifierEnsembleConfig(EnsembleConfig)

Configuration for classifier ensemble.

**Parameters:**

* **n_estimators** *(int)* - Number of estimators.
* **categorical_features_indices** *(List[int])* - Indices of categorical features.
* **softmax_temperature** *(float)* - Softmax temperature.
* **balance_probabilities** *(bool)* - Whether to balance probabilities.
* **average_before_softmax** *(bool)* - Whether to average before softmax.


.. code-block:: python

    def create_inference_engine(model_path, device, config)

Creates an inference engine for TabPFN models.

**Parameters:**

* **model_path** *(str | Path)* - Path to model checkpoint.
* **device** *(torch.device)* - Device for computation.
* **config** - Model configuration.

**Returns:**

* **InferenceEngine** - Created inference engine.


.. code-block:: python

    def initialize_tabpfn_model(model_path, device, config)

Initializes a TabPFN model.

**Parameters:**

* **model_path** *(str | Path)* - Path to model checkpoint.
* **device** *(torch.device)* - Device for computation.
* **config** - Model configuration.

**Returns:**

* **TabPFNModel** - Initialized model.


.. code-block:: python

    def determine_precision(device, inference_precision)

Determines the precision for inference.

**Parameters:**

* **device** *(torch.device)* - Device for computation.
* **inference_precision** *(dtype | str)* - Inference precision specification.

**Returns:**

* **dtype** - Determined precision.


**References:**

Hollmann, N., Muller, S., Purucker, L. et al. Accurate predictions on small data with a tabular foundation model. Nature 637, 319-326 (2025). `<https://doi.org/10.1038/s41586-024-08328-6>`_