**TData**
=========

A PyTorch Dataset implementation designed for handling tabular data with numerical and categorical features.


Functions
~~~~~~~~~

.. code-block:: python

    class TData(Dataset)

A PyTorch Dataset implementation for tabular data with numerical and categorical features.

**Parameters:**

* **is_regression** *(bool)* - Flag indicating if the task is regression. Used to determine label processing and metadata.
* **X** *(Tuple[Optional[Tensor], Optional[Tensor]])* - Tuple containing numerical and categorical features. 
  - `X[0]`: Numerical features (shape: `[n_samples, n_num_features]`).
  - `X[1]`: Categorical features (shape: `[n_samples, n_cat_features]`).
* **Y** *(Dict[str, Tensor])* - Dictionary with labels for train/val/test splits. Keys must include `part`.
* **y_info** *(Dict[str, Any])* - Metadata about the labels. For regression, typically contains `mean` and `std` for de-normalization.
* **part** *(str)* - Data split to use. Must be one of `['train', 'val', 'test']`.


.. code-block:: python

    def get_dim_in(self) -> int

Returns the input feature dimension.

**Returns:**

* **int** - Number of numerical features (`self.X_num.shape[1]`), or 0 if `self.X_num` is `None`.


.. code-block:: python

    def get_categories(self) -> Optional[List[int]]

Returns categorical feature cardinality information.

**Returns:**

* **Optional[List[int]]** - List where each element is the number of unique categories for a categorical feature. Returns `None` if `self.X_cat` is `None`.


.. code-block:: python

    def __len__(self) -> int

Returns the number of samples in the dataset.

**Returns:**

* **int** - Number of samples in the dataset (`len(self.Y)`).


.. code-block:: python

    def __getitem__(self, i) -> Tuple[Union[Tensor, Tuple[Tensor, Tensor]], Tensor]

Retrieves a data sample and its label.

**Parameters:**

* **i** *(int)* - Sample index.

**Returns:**

Returns ``(features, target)``. When both feature types are present,
``features`` is the pair ``(numerical, categorical)``. When only one is
present, ``features`` is that tensor.
