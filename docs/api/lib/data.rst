**Data**
========

A collection of functions for loading, preprocessing, and preparing tabular data for machine learning tasks, including handling missing values, encoding features, and creating data loaders.


Functions
~~~~~~~~~

.. code-block:: python

    def dataname_to_numpy(dataset_name, dataset_path) -> Tuple[ArrayDict, ArrayDict, ArrayDict, Dict[str, Any]]

Loads tabular data from numpy files, including numerical features (N), categorical features (C), labels (y), and dataset metadata.

**Parameters:**

* **dataset_name** *(str)* - Name of the dataset.
* **dataset_path** *(str)* - Path to the dataset directory.

**Returns:**

* **Tuple** containing:
  - Numerical features (N) as a dictionary with keys 'train', 'val', 'test' (or None if unavailable).
  - Categorical features (C) as a dictionary with keys 'train', 'val', 'test' (or None if unavailable).
  - Labels (y) as a dictionary with keys 'train', 'val', 'test'.
  - Dataset metadata from 'info.json'.


.. code-block:: python

    def get_dataset(dataset_name, dataset_path) -> Tuple[Tuple[ArrayDict, ArrayDict, ArrayDict], Tuple[ArrayDict, ArrayDict, ArrayDict], Dict[str, Any]]

Splits loaded data into training/validation and test sets.

**Parameters:**

* **dataset_name** *(str)* - Name of the dataset.
* **dataset_path** *(str)* - Path to the dataset directory.

**Returns:**

* **Tuple** containing:
  - Training/validation data (numerical, categorical, labels).
  - Test data (numerical, categorical, labels).
  - Dataset metadata.


.. code-block:: python

    def data_nan_process(N_data, C_data, num_nan_policy, cat_nan_policy, num_new_value=None, imputer=None, cat_new_value=None) -> Tuple[ArrayDict, ArrayDict, Optional[np.ndarray], Optional[SimpleImputer], Optional[str]]

Processes missing values in numerical and categorical features.

**Parameters:**

* **N_data** *(ArrayDict)* - Numerical features (may contain NaNs).
* **C_data** *(ArrayDict)* - Categorical features (may contain NaNs).
* **num_nan_policy** *(str)* - Strategy for numerical NaNs ('mean' or 'median').
* **cat_nan_policy** *(str)* - Strategy for categorical NaNs ('new' or 'most_frequent').
* **num_new_value** *(Optional[np.ndarray])* - Precomputed values to fill numerical NaNs.
* **imputer** *(Optional[SimpleImputer])* - Pre-fit imputer for categorical features.
* **cat_new_value** *(Optional[str])* - Value to fill categorical NaNs (for 'new' policy).

**Returns:**

* **Tuple** containing:
  - Processed numerical features.
  - Processed categorical features.
  - Values used to fill numerical NaNs.
  - Fitted imputer for categorical features (if used).
  - Value used to fill categorical NaNs (if used).


.. code-block:: python

    def num_enc_process(N_data, num_policy, n_bins=2, y_train=None, is_regression=False, encoder=None) -> Tuple[ArrayDict, Optional[Union[PiecewiseLinearEncoding, UnaryEncoding, BinsEncoding, JohnsonEncoding]]]

Encodes numerical features using various strategies (e.g., piecewise linear, unary, bins).

**Parameters:**

* **N_data** *(ArrayDict)* - Numerical features to encode.
* **num_policy** *(str)* - Encoding strategy (e.g., 'Q_PLE' for quantile-based piecewise linear encoding).
* **n_bins** *(int, optional, Default is 2)* - Number of bins for discretization.
* **y_train** *(Optional[np.ndarray])* - Training labels (for target-based encoding).
* **is_regression** *(bool, optional, Default is False)* - Whether the task is regression.
* **encoder** *(Optional)* - Pre-fit encoder (if None, fits a new one).

**Returns:**

* **Tuple** containing:
  - Encoded numerical features.
  - Fitted encoder.


.. code-block:: python

    def data_enc_process(N_data, C_data, cat_policy, y_train=None, ord_encoder=None, mode_values=None, cat_encoder=None) -> Tuple[ArrayDict, ArrayDict, Optional[OrdinalEncoder], Optional[List[int]], Optional[OneHotEncoder]]

Encodes categorical features using various strategies (e.g., one-hot, target encoding) and handles unknown categories.

**Parameters:**

* **N_data** *(ArrayDict)* - Numerical data (or None).
* **C_data** *(ArrayDict)* - Categorical data (or None).
* **cat_policy** *(str)* - Encoding strategy:
  - `indices`: Return ordinal indices without further encoding.
  - `ordinal`: Use ordinal encoding.
  - `ohe`/`tabr_ohe`: One-hot encoding (with `tabr_ohe` for TabR compatibility).
  - `binary`: Binary encoding (from `category_encoders`).
  - `hash`: Hashing encoding (from `category_encoders`).
  - `loo`: Leave-one-out encoding (supervised, from `category_encoders`).
  - `target`: Target encoding (supervised, from `category_encoders`).
  - `catboost`: CatBoost encoding (supervised, from `category_encoders`).
* **y_train** *(Optional[np.ndarray])* - Training labels (for supervised encodings).
* **ord_encoder** *(Optional[OrdinalEncoder])* - Pre-fitted ordinal encoder.
* **mode_values** *(Optional[List[int]])* - Mode values for replacing unknown categories in validation/test sets.
* **cat_encoder** *(Optional)* - Pre-fitted categorical encoder (e.g., `OneHotEncoder`).

**Returns:**

* **Tuple** containing:
  - Processed numerical data (merged with encoded categoricals if applicable).
  - Unused (returns None if categoricals are merged into numerical data).
  - Fitted ordinal encoder.
  - Mode values for unknown categories.
  - Fitted categorical encoder.


.. code-block:: python

    def data_norm_process(N_data, normalization, seed, normalizer=None) -> Tuple[ArrayDict, Optional[TransformerMixin]]

Applies normalization to numerical features.

**Parameters:**

* **N_data** *(ArrayDict)* - Numerical data (or None).
* **normalization** *(str)* - Normalization strategy:
  - `standard`: StandardScaler (mean=0, std=1).
  - `minmax`: MinMaxScaler (scales to [0, 1]).
  - `quantile`: QuantileTransformer (normalizes to Gaussian distribution).
  - `maxabs`: MaxAbsScaler (scales by maximum absolute value).
  - `power`: PowerTransformer (Yeo-Johnson transformation).
  - `robust`: RobustScaler (resistant to outliers).
  - `none`: No normalization.
* **seed** *(int)* - Random seed for reproducibility (used in `QuantileTransformer`).
* **normalizer** *(Optional[TransformerMixin])* - Pre-fitted normalizer.

**Returns:**

* **Tuple** containing:
  - Normalized numerical data.
  - Fitted normalizer.


.. code-block:: python

    def data_label_process(y_data, is_regression, info=None, encoder=None) -> Tuple[ArrayDict, Dict[str, Any], Optional[LabelEncoder]]

Processes labels for regression or classification tasks.

**Parameters:**

* **y_data** *(ArrayDict)* - Label data.
* **is_regression** *(bool)* - Whether the task is regression.
* **info** *(Optional[Dict[str, Any]])* - Precomputed label statistics (mean, std for regression; classes for classification).
* **encoder** *(Optional[LabelEncoder])* - Pre-fitted label encoder (for classification).

**Returns:**

* **Tuple** containing:
  - Processed labels (standardized for regression; encoded for classification).
  - Metadata (mean/std for regression; classes for classification).
  - Fitted label encoder (for classification).


.. code-block:: python

    def data_loader_process(is_regression, X, Y, y_info, device, batch_size, is_train, is_float=False) -> Tuple[ArrayDict, ArrayDict, ArrayDict, DataLoader, DataLoader, Callable] or Tuple[ArrayDict, ArrayDict, ArrayDict, DataLoader, Callable]

Prepares PyTorch DataLoaders for training/validation or test data, with proper type casting and device placement.

**Parameters:**

* **is_regression** *(bool)* - Whether the task is regression (vs. classification).
* **X** *(Tuple[ArrayDict, ArrayDict])* - Tuple of numerical and categorical data (each as `ArrayDict`).
* **Y** *(ArrayDict)* - Label data.
* **y_info** *(Dict[str, Any])* - Metadata about labels (e.g., mean/std for regression).
* **device** *(torch.device)* - Target device (CPU/GPU) for data.
* **batch_size** *(int)* - Batch size for the DataLoader.
* **is_train** *(bool)* - If True, creates training and validation loaders; if False, creates a test loader.
* **is_float** *(bool, optional, Default is False)* - If True, casts data to `float32`; otherwise uses `float64`.

**Returns:**

Training returns ``(X_num, X_cat, Y, train_loader, val_loader, loss_fn)``.
Evaluation returns ``(X_num, X_cat, Y, test_loader, loss_fn)``.


.. code-block:: python

    def to_tensors(data: ArrayDict) -> Dict[str, torch.Tensor]

Converts numpy arrays in an `ArrayDict` to PyTorch tensors.

**Parameters:**

* **data** *(ArrayDict)* - Dictionary with keys like `'train'`, `'val'`, `'test'` and numpy array values.

**Returns:**

* **Dict[str, torch.Tensor]** - Dictionary with the same keys, where numpy arrays are converted to PyTorch tensors.


.. code-block:: python

    def get_categories(X_cat: Optional[Dict[str, torch.Tensor]]) -> Optional[List[int]]

Computes the number of unique categories for each categorical feature.

**Parameters:**

* **X_cat** *(Optional[Dict[str, torch.Tensor]])* - Categorical data (keys: `'train'`, etc.; values: tensors of shape `(n_samples, n_features)`).

**Returns:**

* **Optional[List[int]]** - List where each element is the number of unique categories for the corresponding feature. Returns `None` if `X_cat` is `None`.


.. code-block:: python

    class Dataset

A dataclass for storing tabular dataset information.

**Fields:**

* **N** *(Optional[ArrayDict])* - Numerical features (or None if not available).
* **C** *(Optional[ArrayDict])* - Categorical features (or None if not available).
* **y** *(ArrayDict)* - Labels for all splits.
* **info** *(Dict[str, Any])* - Dataset metadata.

**Properties:**

* **is_binclass** *(bool)* - Whether the task is binary classification.
* **is_multiclass** *(bool)* - Whether the task is multiclass classification.
* **is_regression** *(bool)* - Whether the task is regression.
* **n_num_features** *(int)* - Number of numerical features.
* **n_cat_features** *(int)* - Number of categorical features.
* **n_features** *(int)* - Total number of features. 