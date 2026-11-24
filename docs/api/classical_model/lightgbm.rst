====================================
LightGBM
====================================

LightGBM classical method implementation.

This section contains the LightGBM implementation for classification and regression tasks. LightGBM is a gradient boosting framework that uses tree-based learning algorithms and is designed to be distributed and efficient with the following advantages: faster training speed and higher efficiency, lower memory usage, better accuracy, support of parallel and GPU learning, and capability of handling large-scale data.

.. automodule:: TALENT.model.classical_methods.lightgbm
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: LightGBMMethod
   :noindex:

   LightGBM method for classification and regression tasks using gradient boosting.
   
   **Key Features:**
   
   - Uses LightGBM library for gradient boosting implementation
   - Supports both classification and regression tasks
   - Handles missing values automatically
   - Provides feature importance analysis
   - Supports early stopping to prevent overfitting
   - Efficient implementation with parallel processing
   - GPU acceleration support
   
   **Algorithm:**
   
   LightGBM is a gradient boosting framework that uses tree-based learning algorithms. It is designed to be distributed and efficient with the following advantages: faster training speed and higher efficiency, lower memory usage, better accuracy, support of parallel and GPU learning, and capability of handling large-scale data.

   .. method:: __init__(args, is_regression)
      :noindex:
      
      Initialize the LightGBM method.
      
      **Parameters:**
      
      * **args** (*object*) -- Configuration arguments containing model settings
      * **is_regression** (*bool*) -- Whether the task is regression (True) or classification (False)

   .. method:: construct_model(model_config=None)
      :noindex:
      
      Construct the LightGBM model instance.
      
      **Parameters:**
      
      * **model_config** (*dict, optional*) -- Model configuration parameters for LightGBM
      
      **Model Creation:**
      
      - For classification: creates `LGBMClassifier`
      - For regression: creates `LGBMRegressor`
      - Configures boosting parameters like learning rate, max depth, etc.

   .. method:: fit(data, info, train=True, config=None)
      :noindex:
      
      Train the LightGBM model on the provided data.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **train** (*bool, default=True*) -- Whether to train the model or just load from checkpoint
      * **config** (*dict, optional*) -- Additional configuration parameters
      
      **Returns:**
      
      * **time_cost** (*float*) -- Training time in seconds
      
      **Training Process:**
      
      1. **Data Preprocessing:** Handles missing values, categorical encoding, normalization
      2. **Model Training:** Fits the LightGBM model with gradient boosting
      3. **Model Saving:** Saves the trained model to disk for later use

   .. method:: predict(data, info, model_name)
      :noindex:
      
      Make predictions using the trained LightGBM model.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **model_name** (*str*) -- Name of the model for saving/loading
      
      **Returns:**
      
      A tuple ``(metric_values, metric_names, predictions)``. The predictions contain class probabilities or regression values in the original target units.
      
      **Prediction Process:**
      
      1. **Data Preprocessing:** Applies same preprocessing as training data
      2. **Model Loading:** Loads the trained LightGBM model
      3. **Prediction:** Generates predictions using the gradient boosting model
      4. **Output:** Returns probabilities for classification or values for regression

**Evaluation Metrics:**

- **For regression:** returns MAE, R2, RMSE metrics
- **For classification:** returns Accuracy, Avg_Recall, Avg_Precision, F1, LogLoss, AUC, Brier and ECE

**References:**

``[1] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). Lightgbm: A highly efficient gradient boosting decision tree. Advances in neural information processing systems, 30.`` 