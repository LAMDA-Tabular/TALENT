**RealMLP**
===========

An improved multilayer perceptron (MLP).


Functions
~~~~~~~~~

.. code-block:: python

    def select_from_config(config: Dict, keys: List)

Selects specific keys from a configuration dictionary.

**Parameters:**

* **config** *(Dict)* - Configuration dictionary.
* **keys** *(List)* - List of keys to select.

**Returns:**

* **Dict** - Dictionary containing only the selected keys.


.. code-block:: python

    def adapt_config(config, **kwargs)

Adapts a configuration dictionary with new parameters.

**Parameters:**

* **config** *(Dict)* - Original configuration dictionary.
* **kwargs** - New parameters to add or override.

**Returns:**

* **Dict** - Modified configuration dictionary.


.. code-block:: python

    def serialize(filename: Union[Path, str], obj: Any, compressed: bool = False, use_json: bool = False, use_yaml: bool = False, use_msgpack: bool = False)

Serializes an object to a file using various formats.

**Parameters:**

* **filename** *(Union[Path, str])* - Output file path.
* **obj** *(Any)* - Object to serialize.
* **compressed** *(bool, optional, Default is False)* - Whether to compress the file.
* **use_json** *(bool, optional, Default is False)* - Whether to use JSON format.
* **use_yaml** *(bool, optional, Default is False)* - Whether to use YAML format.
* **use_msgpack** *(bool, optional, Default is False)* - Whether to use MessagePack format.


.. code-block:: python

    def deserialize(filename: Union[Path, str], compressed: bool = False, use_json: bool = False, use_yaml: bool = False, use_msgpack: bool = False)

Deserializes an object from a file.

**Parameters:**

* **filename** *(Union[Path, str])* - Input file path.
* **compressed** *(bool, optional, Default is False)* - Whether the file is compressed.
* **use_json** *(bool, optional, Default is False)* - Whether to use JSON format.
* **use_yaml** *(bool, optional, Default is False)* - Whether to use YAML format.
* **use_msgpack** *(bool, optional, Default is False)* - Whether to use MessagePack format.

**Returns:**

* **Any** - Deserialized object.


.. code-block:: python

    class Timer

Timer class for measuring execution time.

**Methods:**

* **start(self)** - Start the timer.
* **pause(self)** - Pause the timer.
* **get_result_dict(self)** - Get timing results as dictionary.


.. code-block:: python

    class TimePrinter

Context manager for printing execution time.

**Parameters:**

* **desc** *(str)* - Description for the timing operation.

**Usage:**

.. code-block:: python

    with TimePrinter("Training"):
        model.fit(X, y)


.. code-block:: python

    class TabrQuantileTransformer(BaseEstimator, TransformerMixin)

Quantile transformer with noise addition for tabular data.

**Parameters:**

* **noise** *(float, optional, Default is 1e-3)* - Noise level to add.
* **random_state** *(int, optional)* - Random seed.
* **n_quantiles** *(int, optional, Default is 1000)* - Number of quantiles.
* **subsample** *(int, optional, Default is 1_000_000_000)* - Subsample size.
* **output_distribution** *(str, optional, Default is "normal")* - Output distribution type.

**Methods:**

* **fit(self, X, y=None)** - Fit the transformer.
* **transform(self, X, y=None)** - Transform the data.
* **_add_noise(self, X)** - Add noise to the data.


.. code-block:: python

    class ProcessPoolMapper

Process pool mapper for parallel processing.

**Parameters:**

* **n_processes** *(int)* - Number of processes.
* **chunksize** *(int, optional, Default is 1)* - Chunk size for mapping.

**Methods:**

* **map(self, f, args_tuples: List[Tuple])** - Map function over arguments in parallel.


.. code-block:: python

    def extract_params(config: Dict[str, Any], param_configs: List[Union[Tuple[str, Optional[Union[str, List[str]]]], Tuple[str, Optional[Union[str, List[str]]], Any]]]) -> Dict[str, Any]

Extracts parameters from configuration based on parameter configurations.

**Parameters:**

* **config** *(Dict[str, Any])* - Configuration dictionary.
* **param_configs** *(List)* - List of parameter configurations.

**Returns:**

* **Dict[str, Any]** - Extracted parameters.


.. code-block:: python

    def combine_seeds(seed_1: int, seed_2: int) -> int

Combines two seeds into a single seed.

**Parameters:**

* **seed_1** *(int)* - First seed.
* **seed_2** *(int)* - Second seed.

**Returns:**

* **int** - Combined seed.


**References:**

David Holzmüller and Léo Grinsztajn and Ingo Steinwart. **Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data**. arXiv:2407.04491 [cs.LG], 2025. `<https://arxiv.org/abs/2407.04491>`_






