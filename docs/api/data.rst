Data
=====

.. automodule:: TALENT.model.lib.data
   :members:
   :undoc-members:
   :show-inheritance:

Data Structures
---------------

.. class:: Dataset
   :noindex:

   A comprehensive data structure for representing tabular datasets with numerical and categorical features.
   
   **Attributes:**
   
   * **N** (*Optional[ArrayDict]*) -- Numerical features dictionary with 'train', 'val', 'test' keys
   * **C** (*Optional[ArrayDict]*) -- Categorical features dictionary with 'train', 'val', 'test' keys  
   * **y** (*ArrayDict*) -- Target labels dictionary with 'train', 'val', 'test' keys
   * **info** (*Dict[str, Any]*) -- Dataset metadata including task type and feature counts

**Properties:**

   .. property:: is_binclass
      :noindex:
      
      Check if the dataset is for binary classification.
      
      **Returns:**
      
      * **bool** -- True if task_type is 'binclass'
   
   .. property:: is_multiclass
      :noindex:
      
      Check if the dataset is for multi-class classification.
      
      **Returns:**
      
      * **bool** -- True if task_type is 'multiclass'
   
   .. property:: is_regression
      :noindex:
      
      Check if the dataset is for regression.
      
      **Returns:**
      
      * **bool** -- True if task_type is 'regression'
   
   .. property:: n_num_features
      :noindex:
      
      Get the number of numerical features.
      
      **Returns:**
      
      * **int** -- Number of numerical features
   
   .. property:: n_cat_features
      :noindex:
      
      Get the number of categorical features.
      
      **Returns:**
      
      * **int** -- Number of categorical features
   
   .. property:: n_features
      :noindex:
      
      Get the total number of features.
      
      **Returns:**
      
      * **int** -- Total number of features (numerical + categorical)
   
   .. method:: size(part)
      :noindex:
      
      Get the size of a specific dataset partition.
      
      **Parameters:**
      
      * **part** (*str*) -- Dataset partition ('train', 'val', or 'test')
      
      **Returns:**
      
      * **int** -- Number of samples in the specified partition

Data Loading
------------

.. function:: dataname_to_numpy(dataset_name, dataset_path)
   :noindex:

   Load dataset from numpy files stored in the specified directory structure.
   
   **Parameters:**
   
   * **dataset_name** (*str*) -- Name of the dataset directory
   * **dataset_path** (*str*) -- Base path to the dataset directory
   
   **Returns:**
   
   * **Tuple** -- (N, C, y, info) where:
     * N: Numerical features dictionary or None
     * C: Categorical features dictionary or None  
     * y: Target labels dictionary
     * info: Dataset metadata from info.json
   
   **Expected Directory Structure:**
   
   .. code-block:: text
      
      dataset_path/dataset_name/
      ├── N_train.npy (optional)
      ├── N_val.npy (optional)
      ├── N_test.npy (optional)
      ├── C_train.npy (optional)
      ├── C_val.npy (optional)
      ├── C_test.npy (optional)
      ├── y_train.npy
      ├── y_val.npy
      ├── y_test.npy
      └── info.json

.. function:: get_dataset(dataset_name, dataset_path)
   :noindex:

   Load and split dataset into training/validation and test sets.
   
   **Parameters:**
   
   * **dataset_name** (*str*) -- Name of the dataset directory
   * **dataset_path** (*str*) -- Base path to the dataset directory
   
   **Returns:**
   
   * **Tuple** -- (train_val_data, test_data, info) where:
     * train_val_data: Tuple of (N_trainval, C_trainval, y_trainval)
     * test_data: Tuple of (N_test, C_test, y_test)
     * info: Dataset metadata

.. function:: load_json(path)
   :noindex:

   Load and parse a JSON file.
   
   **Parameters:**
   
   * **path** (*str*) -- Path to the JSON file
   
   **Returns:**
   
   * **dict** -- Parsed JSON content

Data Preprocessing
------------------

.. function:: data_nan_process(N_data, C_data, num_nan_policy, cat_nan_policy, num_new_value=None, imputer=None, cat_new_value=None)
   :noindex:

   Handle missing values (NaN) in numerical and categorical features.
   
   **Parameters:**
   
   * **N_data** (*ArrayDict*) -- Numerical features dictionary
   * **C_data** (*ArrayDict*) -- Categorical features dictionary
   * **num_nan_policy** (*str*) -- Policy for numerical NaN values ('mean', 'median')
   * **cat_nan_policy** (*str*) -- Policy for categorical NaN values ('new', 'most_frequent')
   * **num_new_value** (*Optional[np.ndarray]*) -- Pre-computed values for numerical NaN replacement
   * **imputer** (*Optional[SimpleImputer]*) -- Fitted imputer for categorical features
   * **cat_new_value** (*Optional[str]*) -- Value to replace categorical NaN values
   
   **Returns:**
   
   * **Tuple** -- (N, C, num_new_value, imputer, cat_new_value) where:
     * N: Processed numerical features
     * C: Processed categorical features
     * num_new_value: Values used for numerical NaN replacement
     * imputer: Fitted imputer for categorical features
     * cat_new_value: Value used for categorical NaN replacement
   
   **Numerical NaN Policies:**
   
   * **mean**: Replace NaN with mean of the feature
   * **median**: Replace NaN with median of the feature
   
   **Categorical NaN Policies:**
   
   * **new**: Replace NaN with special token '___null___'
   * **most_frequent**: Use the training mode. An entirely missing training column is retained with the imputer's constant fallback.

.. function:: num_enc_process(N_data, num_policy, n_bins=2, y_train=None, is_regression=False, encoder=None)
   :noindex:

   Apply numerical feature encoding/transformation policies.
   
   **Parameters:**
   
   * **N_data** (*ArrayDict*) -- Numerical features dictionary
   * **num_policy** (*str*) -- Numerical encoding policy
   * **n_bins** (*int, optional*) -- Number of bins for discretization. Defaults to 2.
   * **y_train** (*Optional[np.ndarray]*) -- Training labels for supervised encoding
   * **is_regression** (*bool, optional*) -- Whether task is regression. Defaults to False.
   * **encoder** (*Optional[PiecewiseLinearEncoding]*) -- Pre-fitted encoder
   
   **Returns:**
   
   * **Tuple** -- (N_data, encoder) where:
     * N_data: Transformed numerical features
     * encoder: Fitted encoder for future use
   
   **Encoding Policies:**
   
   * **none**: No transformation
   * **Q_PLE**: Quantile-based Piecewise Linear Encoding
   * **T_PLE**: Tree-based Piecewise Linear Encoding
   * **Q_Unary**: Quantile-based Unary Encoding
   * **T_Unary**: Tree-based Unary Encoding
   * **Q_bins**: Quantile-based Bins Encoding
   * **T_bins**: Tree-based Bins Encoding
   * **Q_Johnson**: Quantile-based Johnson Encoding
   * **T_Johnson**: Tree-based Johnson Encoding

.. function:: data_enc_process(N_data, C_data, cat_policy, y_train=None, ord_encoder=None, mode_values=None, cat_encoder=None)
   :noindex:

   Apply categorical feature encoding policies.
   
   **Parameters:**
   
   * **N_data** (*ArrayDict*) -- Numerical features dictionary
   * **C_data** (*ArrayDict*) -- Categorical features dictionary
   * **cat_policy** (*str*) -- Categorical encoding policy
   * **y_train** (*Optional[np.ndarray]*) -- Training labels for supervised encoding
   * **ord_encoder** (*Optional[OrdinalEncoder]*) -- Pre-fitted ordinal encoder
   * **mode_values** (*Optional[List[int]]*) -- Mode values for unknown categories
   * **cat_encoder** (*Optional[OneHotEncoder]*) -- Pre-fitted categorical encoder
   
   **Returns:**
   
   * **Tuple** -- (N_data, C_data, ord_encoder, mode_values, cat_encoder) where:
     * N_data: Updated numerical features (may be combined with categorical)
     * C_data: Encoded categorical features
     * ord_encoder: Fitted ordinal encoder
     * mode_values: Mode values for unknown categories
     * cat_encoder: Fitted categorical encoder
   
   **Encoding Policies:**
   
   * **indices**: Keep as integer indices (no further encoding)
   * **ordinal**: Ordinal encoding
   * **ohe**: One-hot encoding
   * **binary**: Binary encoding
   * **hash**: Hashing encoding
   * **loo**: Exclude the training row's own target from its category mean. Use the fitted category mean for validation and test rows.
   * **target**: Target encoding
   * **catboost**: Encode training rows in input order using earlier targets for the same category and the training prior. Use the fitted category statistics for validation and test rows.
   * **tabr_ohe**: Special one-hot encoding for TabR model

.. function:: data_norm_process(N_data, normalization, seed, normalizer=None)
   :noindex:

   Apply normalization to numerical features.
   
   **Parameters:**
   
   * **N_data** (*ArrayDict*) -- Numerical features dictionary
   * **normalization** (*str*) -- Normalization method
   * **seed** (*int*) -- Random seed for reproducible normalization
   * **normalizer** (*Optional[TransformerMixin]*) -- Pre-fitted normalizer
   
   **Returns:**
   
   * **Tuple** -- (N_data, normalizer) where:
     * N_data: Normalized numerical features
     * normalizer: Fitted normalizer for future use
   
   **Normalization Methods:**
   
   * **none**: No normalization
   * **standard**: StandardScaler (zero mean, unit variance)
   * **minmax**: MinMaxScaler (scale to [0, 1])
   * **quantile**: QuantileTransformer (normal distribution)
   * **maxabs**: MaxAbsScaler (scale by maximum absolute value)
   * **power**: PowerTransformer (Yeo-Johnson transformation)
   * **robust**: RobustScaler (robust to outliers)

Label Processing
----------------

.. function:: data_label_process(y_data, is_regression, info=None, encoder=None)
   :noindex:

   Process target labels for training.
   
   **Parameters:**
   
   * **y_data** (*ArrayDict*) -- Target labels dictionary
   * **is_regression** (*bool*) -- Whether task is regression
   * **info** (*Optional[Dict[str, Any]]*) -- Label processing information
   * **encoder** (*Optional[LabelEncoder]*) -- Pre-fitted label encoder
   
   **Returns:**
   
   * **Tuple** -- (y, info, encoder) where:
     * y: Processed labels
     * info: Label processing information
     * encoder: Fitted label encoder (None for regression)
   
   **Processing:**
   
   * **Regression**: Standardize labels using mean and standard deviation
   * **Classification**: Encode labels as integers using LabelEncoder

Data Loading for Training
-------------------------

.. function:: data_loader_process(is_regression, X, Y, y_info, device, batch_size, is_train, is_float=False)
   :noindex:

   Create PyTorch DataLoaders for training or inference.
   
   **Parameters:**
   
   * **is_regression** (*bool*) -- Whether task is regression
   * **X** (*Tuple[ArrayDict, ArrayDict]*) -- Tuple of (numerical_features, categorical_features)
   * **Y** (*ArrayDict*) -- Target labels
   * **y_info** (*Dict[str, Any]*) -- Label processing information
   * **device** (*torch.device*) -- Device to load data on (CPU/GPU)
   * **batch_size** (*int*) -- Batch size for DataLoader
   * **is_train** (*bool*) -- Whether creating loaders for training
   * **is_float** (*bool, optional*) -- Whether to use float32 precision. Defaults to False.
   
   **Returns:**
   
   * **Tuple** -- For training: (X_num, X_cat, Y, train_loader, val_loader, loss_fn)
   * **Tuple** -- For inference: (X_num, X_cat, Y, test_loader, loss_fn)
   
   **Features:**
   
   * Converts numpy arrays to PyTorch tensors
   * Moves data to specified device (CPU/GPU)
   * Sets appropriate data types (float32/float64)
   * Creates DataLoader with proper batch size and shuffling
   * Returns appropriate loss function

Utility Functions
-----------------

.. function:: to_tensors(data)
   :noindex:

   Convert numpy arrays to PyTorch tensors.
   
   **Parameters:**
   
   * **data** (*ArrayDict*) -- Dictionary of numpy arrays
   
   **Returns:**
   
   * **Dict[str, torch.Tensor]** -- Dictionary of PyTorch tensors

.. function:: get_categories(X_cat)
   :noindex:

   Get the number of unique categories for each categorical feature.
   
   **Parameters:**
   
   * **X_cat** (*Optional[Dict[str, torch.Tensor]]*) -- Categorical features dictionary
   
   **Returns:**
   
   * **Optional[List[int]]** -- List of category counts for each feature, or None if no categorical features

.. function:: raise_unknown(unknown_what, unknown_value)
   :noindex:

   Raise a ValueError for unknown parameter values.
   
   **Parameters:**
   
   * **unknown_what** (*str*) -- Description of the unknown parameter
   * **unknown_value** (*Any*) -- The unknown value that was provided
   
   **Raises:**
   
   * **ValueError** -- With descriptive error message

Constants
---------

.. data:: BINCLASS
   :noindex:
   
   String constant for binary classification task type.

.. data:: MULTICLASS
   :noindex:
   
   String constant for multi-class classification task type.

.. data:: REGRESSION
   :noindex:
   
   String constant for regression task type.

.. data:: ArrayDict
   :noindex:
   
   Type alias for dictionary mapping partition names to numpy arrays.


