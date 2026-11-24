====================================
Base Class
====================================

Base class for classical methods implementation.

This section contains the abstract base class that defines the interface for all classical machine learning methods in TALENT. The base class provides common functionality for data preprocessing, model training, and evaluation.

.. automodule:: TALENT.model.classical_methods.base
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: classical_methods
   :noindex:

   Abstract base class for all classical machine learning methods in TALENT.
   
   **Key Features:**
   
   - Handles common data preprocessing including missing value imputation, categorical encoding, numerical encoding and binning, data normalization, and label processing
   - Provides unified evaluation metrics calculation
   - Supports both regression and classification tasks
   - Automatically handles data format conversion between training and testing
   - Manages model saving and loading functionality
   
   **Abstract Methods:**
   
   - `construct_model()`: Must be implemented by subclasses to create the specific model
   - `fit()`: Must be implemented by subclasses for model training
   - `predict()`: Must be implemented by subclasses for model prediction

   .. method:: __init__(args, is_regression)
      :noindex:
      
      Initialize the classical method.
      
      **Parameters:**
      
      * **args** (*object*) -- Configuration arguments containing model settings
      * **is_regression** (*bool*) -- Whether the task is regression (True) or classification (False)

   .. method:: data_format(is_train=True, N=None, C=None, y=None)
      :noindex:
      
      Format data for training or testing.
      
      **Parameters:**
      
      * **is_train** (*bool, default=True*) -- Whether processing training data or test data
      * **N** (*array-like, optional*) -- Numerical features data
      * **C** (*array-like, optional*) -- Categorical features data
      * **y** (*array-like, optional*) -- Target labels

   .. method:: construct_model(model_config=None)
      :noindex:
      
      Construct the specific model instance.
      
      **Parameters:**
      
      * **model_config** (*dict, optional*) -- Model configuration parameters
      
      **Abstract Method:** Must be implemented by subclasses.

   .. method:: fit(data, info, train=True, config=None)
      :noindex:
      
      Train the model on the provided data.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **train** (*bool, default=True*) -- Whether to train the model or just load from checkpoint
      * **config** (*dict, optional*) -- Additional configuration parameters
      
      The base implementation prepares the input data and calls ``construct_model``.
      Concrete adapters then fit their estimator.

   .. method:: reset_stats_withconfig(config)
      :noindex:
      
      Reset statistics with new configuration.
      
      **Parameters:**
      
      * **config** (*dict*) -- Configuration parameters

   .. method:: metric(predictions, labels, y_info, threshold=None)
      :noindex:
      
      Calculate evaluation metrics.
      
      **Parameters:**
      
      * **predictions** (*array-like*) -- Model predictions
      * **labels** (*array-like*) -- True labels
      * **y_info** (*dict*) -- Label information
      
      **Returns:**
      
      * **vres** (*tuple*) -- Evaluation metrics values
      * **metric_name** (*tuple*) -- Names of the evaluation metrics
      
      **Metrics:**
      
      - For regression: returns MAE, R2, RMSE metrics
      - For classification: returns Accuracy, Avg_Recall, Avg_Precision, F1, LogLoss, AUC, Brier and ECE

**References:**

``[1] TALENT Framework Documentation. Classical Methods Base Class.`` 